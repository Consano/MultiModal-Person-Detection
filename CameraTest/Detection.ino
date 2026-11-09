#include <Arduino.h>
#include <Wire.h>
#include "stm32h7xx_hal.h"

// ===================== Configuration =====================================================
#define STREAM_320X240 1      // 1 = 320x240, 0 = 160x120 (tables come from the ST OV2640 driver)
#define SEND_GRAY 1           // 1 = send Y only (1 byte/pixel), 0 = send full YUV422 (2 bytes/pixel)
#define ENABLE_TFLM 1         // 1 = run the TensorFlow Lite Micro person classifier on the board
#define PERSON_INDEX 1        // output slots of the TFLM person model (see model_settings.h in the example)
#define NOT_PERSON_INDEX 2
#define USE_DMA 1             // 1 = DCMI+DMA capture (falls back to polling if DMA init fails)
#define CLKRC_VALUE 0x00      // sensor clock divider (0x80 bit = double clock). Python can override.
#define DVP_DIV_VALUE 0x08    // reg 0xD3 pixel-clock divider (ST default 0x08, lower = faster)
#define Y_BYTE_OFFSET 0       // 0 = YUYV (Y at even bytes), 1 = UYVY (Y at odd bytes)
#define DEFAULT_POLARITY_COMBO 4   // bit0 VSPOL, bit1 HSPOL, bit2 PCLK rising (4 = validated)
// =========================================================================================

#if ENABLE_TFLM && !SEND_GRAY
#error "ENABLE_TFLM needs SEND_GRAY 1 (the classifier reads the grayscale frame)"
#endif

// Old TFLM versions pass an error reporter to the interpreter; newer ones removed it. Detect which
// one is installed (override by defining TFLM_USE_ERROR_REPORTER to 0 or 1 yourself if this guesses wrong).
#ifndef TFLM_USE_ERROR_REPORTER
#if defined(__has_include)
#if __has_include("tensorflow/lite/micro/micro_error_reporter.h")
#define TFLM_USE_ERROR_REPORTER 1
#else
#define TFLM_USE_ERROR_REPORTER 0
#endif
#else
#define TFLM_USE_ERROR_REPORTER 0
#endif
#endif

#if ENABLE_TFLM
// Library: Chirale_TensorFlowLite (Library Manager). The old Arduino_TensorFlowLite 2.4.0-ALPHA does
// NOT support the Portenta ("unsupported board"). The model files person_detect_model_data.cpp/.h
// (a plain C array holding the trained model) must be copied into this sketch folder.
#include <Chirale_TensorFlowLite.h>
#if TFLM_USE_ERROR_REPORTER
#include "tensorflow/lite/micro/micro_error_reporter.h"
#endif
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#if defined(__has_include)
#if __has_include("tensorflow/lite/version.h")
#include "tensorflow/lite/version.h"      // old TFLM: defines TFLITE_SCHEMA_VERSION; newer versions moved it
#endif
#endif
#include "person_detect_model_data.h"
#ifndef TFLITE_SCHEMA_VERSION
#define TFLITE_SCHEMA_VERSION 3           // flatbuffer schema version used by all current TFLite models
#endif
#endif

#if STREAM_320X240
#define IMG_W 320
#define IMG_H 240
#else
#define IMG_W 160
#define IMG_H 120
#endif

#define RAW_BYTES (IMG_W * IMG_H * 2)      // sensor output is YUV422: 2 bytes per pixel
#define RAW_WORDS (RAW_BYTES / 4)
#if SEND_GRAY
#define TX_BYTES (IMG_W * IMG_H)
#define TX_FORMAT 0
#else
#define TX_BYTES RAW_BYTES
#define TX_FORMAT 1
#endif

uint8_t rawBuf[RAW_BYTES] __attribute__((aligned(32)));   // DMA target
uint8_t txBuf[TX_BYTES] __attribute__((aligned(32)));     // what gets sent over USB

DCMI_HandleTypeDef hdcmi;
DMA_HandleTypeDef hdmaDcmi;
bool dmaOk = false;
bool armed = false;
bool streaming = false;
unsigned long dmaStartMs = 0;

uint32_t nackCount = 0, frameCounter = 0, timeouts = 0, overruns = 0;
uint32_t dmaStartFail = 0, dmaErrors = 0, lastLisr = 0, lastNdtr = 0, lastMeanY = 0;
uint32_t lastRisr = 0, lastSr = 0, lastDmaCr = 0, dmaTimeoutsInRow = 0;
bool fellBackToPolling = false, dmaInitOk = false;
uint8_t currentClk = CLKRC_VALUE;
uint32_t shortFrames = 0, lastShortWords = 0, lastCapMs = 0;
uint8_t pidH = 0, pidL = 0;

// ---------------------------------------------------------------- sensor (SCCB over Wire1)
void writeCamReg(uint8_t reg, uint8_t val) {
  Wire1.beginTransmission(0x30);
  Wire1.write(reg);
  Wire1.write(val);
  if (Wire1.endTransmission() != 0) nackCount++;
}

uint8_t readCamReg(uint8_t reg) {
  Wire1.beginTransmission(0x30);
  Wire1.write(reg);
  Wire1.endTransmission();
  Wire1.requestFrom(0x30, 1);
  return Wire1.available() ? Wire1.read() : 0xFF;
}

void setSensorClock(uint8_t v) { writeCamReg(0xFF, 0x01); writeCamReg(0x11, v); currentClk = v; }
void setDvpDivider(uint8_t v)  { writeCamReg(0xFF, 0x00); writeCamReg(0xD3, v); }

// Register tables taken unchanged from the ST BSP OV2640 driver (ov2640.c).
// Both end with RGB565 output; setup() overrides that to YUV422 afterwards.
const uint8_t OV2640_QVGA_TABLE[][2] = {
  {0xff, 0x00}, {0x2c, 0xff}, {0x2e, 0xdf}, {0xff, 0x01},
  {0x3c, 0x32}, {0x11, 0x00}, {0x09, 0x02}, {0x04, 0xa8},
  {0x13, 0xe5}, {0x14, 0x48}, {0x2c, 0x0c}, {0x33, 0x78},
  {0x3a, 0x33}, {0x3b, 0xfb}, {0x3e, 0x00}, {0x43, 0x11},
  {0x16, 0x10}, {0x4a, 0x81}, {0x21, 0x99}, {0x24, 0x40},
  {0x25, 0x38}, {0x26, 0x82}, {0x5c, 0x00}, {0x63, 0x00},
  {0x46, 0x3f}, {0x0c, 0x3c}, {0x61, 0x70}, {0x62, 0x80},
  {0x7c, 0x05}, {0x20, 0x80}, {0x28, 0x30}, {0x6c, 0x00},
  {0x6d, 0x80}, {0x6e, 0x00}, {0x70, 0x02}, {0x71, 0x94},
  {0x73, 0xc1}, {0x3d, 0x34}, {0x5a, 0x57}, {0x12, 0x00},
  {0x11, 0x00}, {0x17, 0x11}, {0x18, 0x75}, {0x19, 0x01},
  {0x1a, 0x97}, {0x32, 0x36}, {0x03, 0x0f}, {0x37, 0x40},
  {0x4f, 0xbb}, {0x50, 0x9c}, {0x5a, 0x57}, {0x6d, 0x80},
  {0x6d, 0x38}, {0x39, 0x02}, {0x35, 0x88}, {0x22, 0x0a},
  {0x37, 0x40}, {0x23, 0x00}, {0x34, 0xa0}, {0x36, 0x1a},
  {0x06, 0x02}, {0x07, 0xc0}, {0x0d, 0xb7}, {0x0e, 0x01},
  {0x4c, 0x00}, {0xff, 0x00}, {0xe5, 0x7f}, {0xf9, 0xc0},
  {0x41, 0x24}, {0xe0, 0x14}, {0x76, 0xff}, {0x33, 0xa0},
  {0x42, 0x20}, {0x43, 0x18}, {0x4c, 0x00}, {0x87, 0xd0},
  {0x88, 0x3f}, {0xd7, 0x03}, {0xd9, 0x10}, {0xd3, 0x82},
  {0xc8, 0x08}, {0xc9, 0x80}, {0x7d, 0x00}, {0x7c, 0x03},
  {0x7d, 0x48}, {0x7c, 0x08}, {0x7d, 0x20}, {0x7d, 0x10},
  {0x7d, 0x0e}, {0x90, 0x00}, {0x91, 0x0e}, {0x91, 0x1a},
  {0x91, 0x31}, {0x91, 0x5a}, {0x91, 0x69}, {0x91, 0x75},
  {0x91, 0x7e}, {0x91, 0x88}, {0x91, 0x8f}, {0x91, 0x96},
  {0x91, 0xa3}, {0x91, 0xaf}, {0x91, 0xc4}, {0x91, 0xd7},
  {0x91, 0xe8}, {0x91, 0x20}, {0x92, 0x00}, {0x93, 0x06},
  {0x93, 0xe3}, {0x93, 0x02}, {0x93, 0x02}, {0x93, 0x00},
  {0x93, 0x04}, {0x93, 0x00}, {0x93, 0x03}, {0x93, 0x00},
  {0x93, 0x00}, {0x93, 0x00}, {0x93, 0x00}, {0x93, 0x00},
  {0x93, 0x00}, {0x93, 0x00}, {0x96, 0x00}, {0x97, 0x08},
  {0x97, 0x19}, {0x97, 0x02}, {0x97, 0x0c}, {0x97, 0x24},
  {0x97, 0x30}, {0x97, 0x28}, {0x97, 0x26}, {0x97, 0x02},
  {0x97, 0x98}, {0x97, 0x80}, {0x97, 0x00}, {0x97, 0x00},
  {0xc3, 0xef}, {0xff, 0x00}, {0xba, 0xdc}, {0xbb, 0x08},
  {0xb6, 0x24}, {0xb8, 0x33}, {0xb7, 0x20}, {0xb9, 0x30},
  {0xb3, 0xb4}, {0xb4, 0xca}, {0xb5, 0x43}, {0xb0, 0x5c},
  {0xb1, 0x4f}, {0xb2, 0x06}, {0xc7, 0x00}, {0xc6, 0x51},
  {0xc5, 0x11}, {0xc4, 0x9c}, {0xbf, 0x00}, {0xbc, 0x64},
  {0xa6, 0x00}, {0xa7, 0x1e}, {0xa7, 0x6b}, {0xa7, 0x47},
  {0xa7, 0x33}, {0xa7, 0x00}, {0xa7, 0x23}, {0xa7, 0x2e},
  {0xa7, 0x85}, {0xa7, 0x42}, {0xa7, 0x33}, {0xa7, 0x00},
  {0xa7, 0x23}, {0xa7, 0x1b}, {0xa7, 0x74}, {0xa7, 0x42},
  {0xa7, 0x33}, {0xa7, 0x00}, {0xa7, 0x23}, {0xc0, 0xc8},
  {0xc1, 0x96}, {0x8c, 0x00}, {0x86, 0x3d}, {0x50, 0x92},
  {0x51, 0x90}, {0x52, 0x2c}, {0x53, 0x00}, {0x54, 0x00},
  {0x55, 0x88}, {0x5a, 0x50}, {0x5b, 0x3c}, {0x5c, 0x00},
  {0xd3, 0x04}, {0x7f, 0x00}, {0xda, 0x00}, {0xe5, 0x1f},
  {0xe1, 0x67}, {0xe0, 0x00}, {0xdd, 0x7f}, {0x05, 0x00},
  {0xff, 0x00}, {0xe0, 0x04}, {0xc0, 0xc8}, {0xc1, 0x96},
  {0x86, 0x3d}, {0x50, 0x92}, {0x51, 0x90}, {0x52, 0x2c},
  {0x53, 0x00}, {0x54, 0x00}, {0x55, 0x88}, {0x57, 0x00},
  {0x5a, 0x50}, {0x5b, 0x3c}, {0x5c, 0x00}, {0xd3, 0x08},
  {0xe0, 0x00}, {0xff, 0x00}, {0x05, 0x00}, {0xda, 0x08},
  {0xda, 0x09}, {0x98, 0x00}, {0x99, 0x00}, {0x00, 0x00},
};

const uint8_t OV2640_QQVGA_TABLE[][2] = {
  {0xff, 0x00}, {0x2c, 0xff}, {0x2e, 0xdf}, {0xff, 0x01},
  {0x3c, 0x32}, {0x11, 0x00}, {0x09, 0x02}, {0x04, 0xa8},
  {0x13, 0xe5}, {0x14, 0x48}, {0x2c, 0x0c}, {0x33, 0x78},
  {0x3a, 0x33}, {0x3b, 0xfb}, {0x3e, 0x00}, {0x43, 0x11},
  {0x16, 0x10}, {0x4a, 0x81}, {0x21, 0x99}, {0x24, 0x40},
  {0x25, 0x38}, {0x26, 0x82}, {0x5c, 0x00}, {0x63, 0x00},
  {0x46, 0x3f}, {0x0c, 0x3c}, {0x61, 0x70}, {0x62, 0x80},
  {0x7c, 0x05}, {0x20, 0x80}, {0x28, 0x30}, {0x6c, 0x00},
  {0x6d, 0x80}, {0x6e, 0x00}, {0x70, 0x02}, {0x71, 0x94},
  {0x73, 0xc1}, {0x3d, 0x34}, {0x5a, 0x57}, {0x12, 0x00},
  {0x11, 0x00}, {0x17, 0x11}, {0x18, 0x75}, {0x19, 0x01},
  {0x1a, 0x97}, {0x32, 0x36}, {0x03, 0x0f}, {0x37, 0x40},
  {0x4f, 0xbb}, {0x50, 0x9c}, {0x5a, 0x57}, {0x6d, 0x80},
  {0x6d, 0x38}, {0x39, 0x02}, {0x35, 0x88}, {0x22, 0x0a},
  {0x37, 0x40}, {0x23, 0x00}, {0x34, 0xa0}, {0x36, 0x1a},
  {0x06, 0x02}, {0x07, 0xc0}, {0x0d, 0xb7}, {0x0e, 0x01},
  {0x4c, 0x00}, {0xff, 0x00}, {0xe5, 0x7f}, {0xf9, 0xc0},
  {0x41, 0x24}, {0xe0, 0x14}, {0x76, 0xff}, {0x33, 0xa0},
  {0x42, 0x20}, {0x43, 0x18}, {0x4c, 0x00}, {0x87, 0xd0},
  {0x88, 0x3f}, {0xd7, 0x03}, {0xd9, 0x10}, {0xd3, 0x82},
  {0xc8, 0x08}, {0xc9, 0x80}, {0x7d, 0x00}, {0x7c, 0x03},
  {0x7d, 0x48}, {0x7c, 0x08}, {0x7d, 0x20}, {0x7d, 0x10},
  {0x7d, 0x0e}, {0x90, 0x00}, {0x91, 0x0e}, {0x91, 0x1a},
  {0x91, 0x31}, {0x91, 0x5a}, {0x91, 0x69}, {0x91, 0x75},
  {0x91, 0x7e}, {0x91, 0x88}, {0x91, 0x8f}, {0x91, 0x96},
  {0x91, 0xa3}, {0x91, 0xaf}, {0x91, 0xc4}, {0x91, 0xd7},
  {0x91, 0xe8}, {0x91, 0x20}, {0x92, 0x00}, {0x93, 0x06},
  {0x93, 0xe3}, {0x93, 0x02}, {0x93, 0x02}, {0x93, 0x00},
  {0x93, 0x04}, {0x93, 0x00}, {0x93, 0x03}, {0x93, 0x00},
  {0x93, 0x00}, {0x93, 0x00}, {0x93, 0x00}, {0x93, 0x00},
  {0x93, 0x00}, {0x93, 0x00}, {0x96, 0x00}, {0x97, 0x08},
  {0x97, 0x19}, {0x97, 0x02}, {0x97, 0x0c}, {0x97, 0x24},
  {0x97, 0x30}, {0x97, 0x28}, {0x97, 0x26}, {0x97, 0x02},
  {0x97, 0x98}, {0x97, 0x80}, {0x97, 0x00}, {0x97, 0x00},
  {0xc3, 0xef}, {0xff, 0x00}, {0xba, 0xdc}, {0xbb, 0x08},
  {0xb6, 0x24}, {0xb8, 0x33}, {0xb7, 0x20}, {0xb9, 0x30},
  {0xb3, 0xb4}, {0xb4, 0xca}, {0xb5, 0x43}, {0xb0, 0x5c},
  {0xb1, 0x4f}, {0xb2, 0x06}, {0xc7, 0x00}, {0xc6, 0x51},
  {0xc5, 0x11}, {0xc4, 0x9c}, {0xbf, 0x00}, {0xbc, 0x64},
  {0xa6, 0x00}, {0xa7, 0x1e}, {0xa7, 0x6b}, {0xa7, 0x47},
  {0xa7, 0x33}, {0xa7, 0x00}, {0xa7, 0x23}, {0xa7, 0x2e},
  {0xa7, 0x85}, {0xa7, 0x42}, {0xa7, 0x33}, {0xa7, 0x00},
  {0xa7, 0x23}, {0xa7, 0x1b}, {0xa7, 0x74}, {0xa7, 0x42},
  {0xa7, 0x33}, {0xa7, 0x00}, {0xa7, 0x23}, {0xc0, 0xc8},
  {0xc1, 0x96}, {0x8c, 0x00}, {0x86, 0x3d}, {0x50, 0x92},
  {0x51, 0x90}, {0x52, 0x2c}, {0x53, 0x00}, {0x54, 0x00},
  {0x55, 0x88}, {0x5a, 0x50}, {0x5b, 0x3c}, {0x5c, 0x00},
  {0xd3, 0x04}, {0x7f, 0x00}, {0xda, 0x00}, {0xe5, 0x1f},
  {0xe1, 0x67}, {0xe0, 0x00}, {0xdd, 0x7f}, {0x05, 0x00},
  {0xff, 0x00}, {0xe0, 0x04}, {0xc0, 0xc8}, {0xc1, 0x96},
  {0x86, 0x3d}, {0x50, 0x92}, {0x51, 0x90}, {0x52, 0x2c},
  {0x53, 0x00}, {0x54, 0x00}, {0x55, 0x88}, {0x57, 0x00},
  {0x5a, 0x28}, {0x5b, 0x1e}, {0x5c, 0x00}, {0xd3, 0x08},
  {0xe0, 0x00}, {0xff, 0x00}, {0x05, 0x00}, {0xda, 0x08},
  {0xda, 0x09}, {0x98, 0x00}, {0x99, 0x00}, {0x00, 0x00},
};

// ---------------------------------------------------------------- DCMI + DMA
// combo bits: 0 = VSPOL, 1 = HSPOL, 2 = PCLK rising. The polarity bits give the level during
// blanking; this sensor setup has blanking low for both syncs, so VS=0, HS=0.
void setPolarity(uint8_t combo) {
  DCMI->CR &= ~(DCMI_CR_CAPTURE | DCMI_CR_ENABLE);
  uint32_t cr = DCMI->CR & ~(DCMI_CR_VSPOL | DCMI_CR_HSPOL | DCMI_CR_PCKPOL);
  if (combo & 1) cr |= DCMI_CR_VSPOL;
  if (combo & 2) cr |= DCMI_CR_HSPOL;
  if (combo & 4) cr |= DCMI_CR_PCKPOL;
  DCMI->CR = cr;
}

void initDCMIHardware() {
  __HAL_RCC_DCMI_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOI_CLK_ENABLE();

  GPIO_InitTypeDef GPIO_InitStruct = {0};
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF13_DCMI;

  GPIO_InitStruct.Pin = GPIO_PIN_4 | GPIO_PIN_6;                       // PA4 HREF, PA6 PCLK
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
  GPIO_InitStruct.Pin = GPIO_PIN_4 | GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7;   // PI5 VSYNC, PI4/6/7 D5/D6/D7
  HAL_GPIO_Init(GPIOI, &GPIO_InitStruct);
  GPIO_InitStruct.Pin = GPIO_PIN_9 | GPIO_PIN_10 | GPIO_PIN_11 | GPIO_PIN_12 | GPIO_PIN_14;  // D0..D4
  HAL_GPIO_Init(GPIOH, &GPIO_InitStruct);

  hdcmi.Instance = DCMI;
  hdcmi.Init.SynchroMode = DCMI_SYNCHRO_HARDWARE;
  hdcmi.Init.PCKPolarity = DCMI_PCKPOLARITY_RISING;
  hdcmi.Init.VSPolarity = DCMI_VSPOLARITY_LOW;
  hdcmi.Init.HSPolarity = DCMI_HSPOLARITY_LOW;
  hdcmi.Init.CaptureRate = DCMI_CR_ALL_FRAME;
  hdcmi.Init.ExtendedDataMode = DCMI_EXTEND_DATA_8B;
  hdcmi.Init.JPEGMode = DCMI_JPEG_DISABLE;
  hdcmi.Init.ByteSelectMode = DCMI_BSM_ALL;
  hdcmi.Init.ByteSelectStart = DCMI_OEBS_ODD;
  hdcmi.Init.LineSelectMode = DCMI_LSM_ALL;
  hdcmi.Init.LineSelectStart = DCMI_OELS_ODD;
  HAL_DCMI_Init(&hdcmi);

  setPolarity(DEFAULT_POLARITY_COMBO);
}

void initDMA() {
#if USE_DMA
  __HAL_RCC_DMA2_CLK_ENABLE();
  hdmaDcmi.Instance = DMA2_Stream1;
  hdmaDcmi.Init.Request = DMA_REQUEST_DCMI;
  hdmaDcmi.Init.Direction = DMA_PERIPH_TO_MEMORY;
  hdmaDcmi.Init.PeriphInc = DMA_PINC_DISABLE;
  hdmaDcmi.Init.MemInc = DMA_MINC_ENABLE;
  hdmaDcmi.Init.PeriphDataAlignment = DMA_PDATAALIGN_WORD;
  hdmaDcmi.Init.MemDataAlignment = DMA_MDATAALIGN_WORD;
  hdmaDcmi.Init.Mode = DMA_NORMAL;
  hdmaDcmi.Init.Priority = DMA_PRIORITY_HIGH;
  hdmaDcmi.Init.FIFOMode = DMA_FIFOMODE_ENABLE;
  hdmaDcmi.Init.FIFOThreshold = DMA_FIFO_THRESHOLD_FULL;
  hdmaDcmi.Init.MemBurst = DMA_MBURST_INC4;
  hdmaDcmi.Init.PeriphBurst = DMA_PBURST_SINGLE;
  dmaOk = (HAL_DMA_Init(&hdmaDcmi) == HAL_OK);
  dmaInitOk = dmaOk;
#endif
}

// Start capturing the NEXT frame into rawBuf (DCMI waits for the next VSYNC frame start).
bool armDMA() {
  DCMI->CR &= ~(DCMI_CR_CAPTURE | DCMI_CR_ENABLE);
  DCMI->ICR = 0x1F;
  // Program the stream registers directly. HAL_DMA_Start refuses every start after the first
  // (measured: millions of failed starts) because the handle stays locked/busy when the HAL's
  // own IRQ/poll completion paths are not used. HAL_DMA_Init already set the stream's mode,
  // data sizes, FIFO and DMAMUX request; only address and length change per frame.
  DMA_Stream_TypeDef *s = (DMA_Stream_TypeDef *)hdmaDcmi.Instance;
  s->CR &= ~DMA_SxCR_EN;
  uint32_t guard = 1000000;
  while ((s->CR & DMA_SxCR_EN) && guard--) {}
  if (s->CR & DMA_SxCR_EN) { dmaStartFail++; armed = false; return false; }
  DMA2->LIFCR = 0x0F40;                          // clear all stream-1 flags (FE, DME, TE, HT, TC)
  s->PAR = (uint32_t)(uintptr_t)&DCMI->DR;
  s->M0AR = (uint32_t)(uintptr_t)rawBuf;
  s->NDTR = RAW_WORDS;
  s->CR |= DMA_SxCR_EN;
  DCMI->CR |= DCMI_CR_ENABLE;
  DCMI->CR |= DCMI_CR_CM | DCMI_CR_CAPTURE;      // snapshot mode, start at next VSYNC
  dmaStartMs = millis();
  armed = true;
  return true;
}

void abortDMA() {
  DCMI->CR &= ~(DCMI_CR_CAPTURE | DCMI_CR_ENABLE);
  DMA_Stream_TypeDef *s = (DMA_Stream_TypeDef *)hdmaDcmi.Instance;
  s->CR &= ~DMA_SxCR_EN;
  while (s->CR & DMA_SxCR_EN) {}
  armed = false;
}

// Wait for the armed DMA transfer to finish. false = timeout (DMA aborted).
bool waitDMA(bool &overrun) {
  DMA_Stream_TypeDef *s = (DMA_Stream_TypeDef *)hdmaDcmi.Instance;
  while (s->CR & DMA_SxCR_EN) {                  // EN clears itself when the transfer completes
    if (millis() - dmaStartMs > 1000) {
      lastNdtr = s->NDTR; lastLisr = DMA2->LISR;   // how far it got before we gave up
      lastRisr = DCMI->RISR; lastSr = DCMI->SR; lastDmaCr = s->CR;
      abortDMA();
      return false;
    }
  }
  if (s->NDTR != 0) {                            // EN also clears on a DMA error: not a real frame
    dmaErrors++;
    lastNdtr = s->NDTR; lastLisr = DMA2->LISR;
    armed = false;
    return false;
  }
  overrun = (DCMI->RISR & 0x02) != 0;
  armed = false;
  SCB_InvalidateDCache_by_Addr((uint32_t *)rawBuf, RAW_BYTES);   // CPU must not see stale cache
  return true;
}

// Fallback: CPU polling capture into rawBuf (slow pixel clocks only).
bool capturePolling(bool &overrun) {
  uint32_t *dest = (uint32_t *)rawBuf;
  uint32_t n = 0, idleSpins = 0;
  DCMI->CR &= ~(DCMI_CR_CAPTURE | DCMI_CR_ENABLE);
  DCMI->ICR = 0x1F;
  DCMI->CR |= DCMI_CR_ENABLE;
  DCMI->CR |= DCMI_CR_CM | DCMI_CR_CAPTURE;
  unsigned long t0 = millis();
  while (n < RAW_WORDS) {
    if (DCMI->SR & DCMI_SR_FNE) {
      dest[n++] = DCMI->DR;
    } else {
      uint32_t ris = DCMI->RISR;
      if (ris & 0x02) break;                                   // FIFO overrun: frame is corrupt
      if ((ris & 0x01) && !(DCMI->SR & DCMI_SR_FNE)) break;    // frame ended, FIFO empty, too few words
      if ((++idleSpins & 0xFFFF) == 0 && millis() - t0 > 3000) break;
    }
  }
  bool full = (n == RAW_WORDS);
  overrun = false;
  DCMI->CR &= ~DCMI_CR_CAPTURE;
  if (!full) { shortFrames++; lastShortWords = n; }            // dropped immediately, no long wait
  return full;
}

// ---------------------------------------------------------------- frame output
void prepareTx() {
#if SEND_GRAY
  const uint32_t *src = (const uint32_t *)rawBuf;
  uint16_t *dst = (uint16_t *)txBuf;
  uint32_t sum = 0;
  for (uint32_t i = 0; i < RAW_WORDS; i++) {
    uint32_t w = src[i];                          // bytes: [b0 b1 b2 b3] = Y0 U Y1 V (YUYV)
#if Y_BYTE_OFFSET == 0
    uint32_t y0 = w & 0xFF, y1 = (w >> 16) & 0xFF;
#else
    uint32_t y0 = (w >> 8) & 0xFF, y1 = (w >> 24) & 0xFF;
#endif
    dst[i] = (uint16_t)(y0 | (y1 << 8));
    sum += y0 + y1;
  }
  lastMeanY = sum / (IMG_W * IMG_H);
#else
  memcpy(txBuf, rawBuf, RAW_BYTES);
#endif
}

// ---------------------------------------------------------------- on-device person classifier
// Result of the last inference, sent with every frame. personPct/notPersonPct are 0..100.
int modelStatus = -1;            // -1 not initialised, 0 ready, >0 setup/runtime error (see initModel)
bool modelRanThisFrame = false;
uint8_t personPct = 0, notPersonPct = 0;
uint32_t inferMs = 0;

#if ENABLE_TFLM
constexpr int kModelSide = 96;                   // the TFLM person model takes 96x96 grayscale
constexpr int kTensorArenaSize = 144 * 1024;     // 136 KB in the example; a little extra headroom
static uint8_t tensorArena[kTensorArenaSize] __attribute__((aligned(16)));
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *modelInput = nullptr;
static bool modelIsInt8 = true;                  // older example models are uint8, newer ones int8
#if TFLM_USE_ERROR_REPORTER
static tflite::ErrorReporter *errorReporter = nullptr;
#endif

void initModel() {
#if TFLM_USE_ERROR_REPORTER
  static tflite::MicroErrorReporter microErrorReporter;
  errorReporter = &microErrorReporter;
#endif
  const tflite::Model *model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) { modelStatus = 1; return; }

  static tflite::MicroMutableOpResolver<5> resolver;      // the ops the person model uses
  resolver.AddAveragePool2D();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddReshape();
  resolver.AddSoftmax();

#if TFLM_USE_ERROR_REPORTER
  static tflite::MicroInterpreter staticInterpreter(model, resolver, tensorArena, kTensorArenaSize, errorReporter);
#else
  static tflite::MicroInterpreter staticInterpreter(model, resolver, tensorArena, kTensorArenaSize);
#endif
  interpreter = &staticInterpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) { modelStatus = 2; return; }   // arena too small?

  modelInput = interpreter->input(0);
  modelIsInt8 = (modelInput->type == kTfLiteInt8);
  if ((modelInput->type != kTfLiteInt8 && modelInput->type != kTfLiteUInt8) || modelInput->dims->size != 4 ||
      modelInput->dims->data[1] != kModelSide || modelInput->dims->data[2] != kModelSide ||
      modelInput->dims->data[3] != 1) { modelStatus = 3; return; }                // not the model we expect
  modelStatus = 0;
}

// Centre-crop the largest square from the grayscale frame (txBuf), scale it to 96x96 with a
// 2x2 box average, and shift to the signed range the int8 model expects.
void prepareModelInput() {
  const uint8_t *g = (const uint8_t *)txBuf;
  const int side = (IMG_W < IMG_H) ? IMG_W : IMG_H;
  const int x0 = (IMG_W - side) / 2, y0 = (IMG_H - side) / 2;
  const int step = (side * 256) / kModelSide;              // source pixels per output pixel, 8.8 fixed point
  for (int y = 0; y < kModelSide; y++) {
    int sy = y0 + ((y * step + step / 2) >> 8);
    if (sy > IMG_H - 2) sy = IMG_H - 2;
    for (int x = 0; x < kModelSide; x++) {
      int sx = x0 + ((x * step + step / 2) >> 8);
      if (sx > IMG_W - 2) sx = IMG_W - 2;
      const uint8_t *p = g + sy * IMG_W + sx;
      int v = (p[0] + p[1] + p[IMG_W] + p[IMG_W + 1]) >> 2;
      if (modelIsInt8) modelInput->data.int8[y * kModelSide + x] = (int8_t)(v - 128);
      else             modelInput->data.uint8[y * kModelSide + x] = (uint8_t)v;
    }
  }
}

void runModel() {
  modelRanThisFrame = false;
  if (modelStatus != 0) return;
  prepareModelInput();
  unsigned long t0 = millis();
  if (interpreter->Invoke() != kTfLiteOk) { modelStatus = 4; return; }
  inferMs = millis() - t0;
  TfLiteTensor *out = interpreter->output(0);
  int p, n;
  if (out->type == kTfLiteInt8) {                          // int8 softmax output: -128..127 -> 0..100 %
    p = out->data.int8[PERSON_INDEX] + 128; n = out->data.int8[NOT_PERSON_INDEX] + 128;
  } else if (out->type == kTfLiteUInt8) {                  // uint8 softmax output: 0..255 -> 0..100 %
    p = out->data.uint8[PERSON_INDEX]; n = out->data.uint8[NOT_PERSON_INDEX];
  } else { modelStatus = 5; return; }
  personPct = (uint8_t)((p * 100) / 255);
  notPersonPct = (uint8_t)((n * 100) / 255);
  modelRanThisFrame = true;
}
#else
void initModel() {}
void runModel() {}
#endif

struct __attribute__((packed)) FrameHeader {
  char magic[4];       // "FRMX"
  uint32_t counter;
  uint16_t w, h;
  uint8_t format;      // 0 = gray8, 1 = YUV422 (YUYV)
  uint8_t flags;       // bit0 = FIFO overrun, bit1 = classifier result in this header is valid
  uint32_t length;
  uint8_t personPct;     // classifier "person" score, 0..100
  uint8_t notPersonPct;  // classifier "not a person" score, 0..100
  uint16_t inferMs;      // inference time on the board
};

void sendFrame(bool overrun) {
  FrameHeader hdr = {{'F', 'R', 'M', 'X'}, ++frameCounter, IMG_W, IMG_H, TX_FORMAT,
                     (uint8_t)((overrun ? 1 : 0) | (modelRanThisFrame ? 2 : 0)), TX_BYTES,
                     personPct, notPersonPct, (uint16_t)inferMs};
  Serial.write((const uint8_t *)&hdr, sizeof(hdr));
  Serial.write(txBuf, TX_BYTES);
}

void streamOnce() {
  bool overrun = false;
  if (dmaOk) {
    if (!armed && !armDMA()) return;              // start failure is counted in dmaStartFail
    if (!waitDMA(overrun)) {
      timeouts++;
      // DMA never completed (twice in a row): give up on it and use CPU polling with a slow
      // sensor clock, which is known to work on this hardware. The 'i' line reports fb=1.
      if (++dmaTimeoutsInRow >= 2) {
        dmaOk = false;
        fellBackToPolling = true;
        setSensorClock(0x07);
      }
      return;
    }
    dmaTimeoutsInRow = 0;
    lastCapMs = millis() - dmaStartMs;
    prepareTx();
    if (!armDMA() && ++dmaTimeoutsInRow >= 2) {   // capture the next frame while this one is sent
      dmaOk = false; fellBackToPolling = true; setSensorClock(0x07);
    }
  } else {
    unsigned long tCap = millis();
    if (!capturePolling(overrun)) { timeouts++; return; }
    lastCapMs = millis() - tCap;
    prepareTx();
  }
  if (overrun) overruns++;
  runModel();                                     // capture of the next frame is already running
  sendFrame(overrun);
}

// ---------------------------------------------------------------- serial commands
int hexVal(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

bool readHexByte(uint8_t &out) {
  int v = 0;
  for (int k = 0; k < 2; k++) {
    unsigned long t = millis();
    while (!Serial.available() && millis() - t < 100) {}
    if (!Serial.available()) return false;
    int d = hexVal(Serial.read());
    if (d < 0) return false;
    v = (v << 4) | d;
  }
  out = (uint8_t)v;
  return true;
}

// Commands: s = start stream, x = stop, i = info line, 0..7 = polarity combo,
//           k<2 hex> = sensor CLKRC, v<2 hex> = DVP divider (reg 0xD3), r = retry DMA
void handleSerial() {
  while (Serial.available()) {
    char cmd = Serial.read();
    uint8_t v;
    if (cmd == 's') {
      streaming = true;
    } else if (cmd == 'x') {
      streaming = false;
      if (armed) abortDMA();
    } else if (cmd >= '0' && cmd <= '7') {
      if (armed) abortDMA();
      setPolarity(cmd - '0');
    } else if (cmd == 'k') {
      if (readHexByte(v)) {
        if (!dmaOk && (v & 0x3F) < 0x07) v = 0x07;   // CPU polling cannot keep up with faster clocks
        setSensorClock(v);
      }
    } else if (cmd == 'v') {
      if (readHexByte(v)) setDvpDivider(v);
    } else if (cmd == 'r') {                          // retry DMA (leaves polling fallback)
      if (armed) abortDMA();
      streaming = false;
      dmaOk = dmaInitOk; fellBackToPolling = false; dmaTimeoutsInRow = 0;
    } else if (cmd == 'i') {
      // Raw sensor exposure/gain registers (bank 1): if exposure keeps rising while the stream
      // slows down, auto-exposure is stretching the frame time. Raw values, unverified map.
      writeCamReg(0xFF, 0x01);
      uint8_t r04 = readCamReg(0x04), r10 = readCamReg(0x10), r45 = readCamReg(0x45), r00 = readCamReg(0x00);
      char buf[640];
      snprintf(buf, sizeof(buf),
               "CAMID %02X %02X NACKS %lu | %dx%d fmt=%d | dma=%d raw=0x%08lX tx=0x%08lX | frames=%lu timeouts=%lu overruns=%lu | dmaStartFail=%lu dmaErr=%lu ndtr=%lu lisr=0x%08lX meanY=%lu | fb=%d risr=0x%02lX sr=0x%02lX dmacr=0x%08lX | clk=0x%02X short=%lu lastShort=%lu | cap=%lums r04=%02X r10=%02X r45=%02X r00=%02X | model=%d person=%u%% notp=%u%% infer=%lums",
               pidH, pidL, (unsigned long)nackCount, IMG_W, IMG_H, TX_FORMAT, dmaOk ? 1 : 0,
               (unsigned long)rawBuf, (unsigned long)txBuf,
               (unsigned long)frameCounter, (unsigned long)timeouts, (unsigned long)overruns,
               (unsigned long)dmaStartFail, (unsigned long)dmaErrors,
               (unsigned long)lastNdtr, (unsigned long)lastLisr, (unsigned long)lastMeanY,
               fellBackToPolling ? 1 : 0, (unsigned long)lastRisr, (unsigned long)lastSr,
               (unsigned long)lastDmaCr, currentClk, (unsigned long)shortFrames, (unsigned long)lastShortWords,
               (unsigned long)lastCapMs, r04, r10, r45, r00,
               modelStatus, (unsigned)personPct, (unsigned)notPersonPct, (unsigned long)inferMs);
      Serial.println(buf);
      Serial.flush();
    }
  }
}

void setup() {
  Serial.begin(921600);
  Wire1.begin();
  Wire1.setClock(100000);

  // Same preparation as the ST driver: select sensor bank, software reset, wait
  writeCamReg(0xFF, 0x01);
  writeCamReg(0x12, 0x80);
  delay(200);

#if STREAM_320X240
  const uint8_t (*table)[2] = OV2640_QVGA_TABLE;
  const int numRegs = sizeof(OV2640_QVGA_TABLE) / sizeof(OV2640_QVGA_TABLE[0]);
#else
  const uint8_t (*table)[2] = OV2640_QQVGA_TABLE;
  const int numRegs = sizeof(OV2640_QQVGA_TABLE) / sizeof(OV2640_QQVGA_TABLE[0]);
#endif
  for (int i = 0; i < numRegs; i++) {
    writeCamReg(table[i][0], table[i][1]);
    delay(1);
  }

  writeCamReg(0xFF, 0x00);
  writeCamReg(0xDA, 0x00);                        // uncompressed YUV422 (table ends in RGB565)
  setDvpDivider(DVP_DIV_VALUE);
  setSensorClock(CLKRC_VALUE);
  delay(200);

  writeCamReg(0xFF, 0x01);
  pidH = readCamReg(0x0A);                        // expect 0x26
  pidL = readCamReg(0x0B);                        // expect 0x42 (some modules report 0x41)

  initDCMIHardware();
  initDMA();
  initModel();                                    // modelStatus: 0 = ready (see initModel)
}

void loop() {
  handleSerial();
  if (streaming) streamOnce();
}