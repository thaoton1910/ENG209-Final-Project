#include <Arduino.h>
#include <driver/i2s.h>

// =====================================================
// AURA BAND DSP FILTER TEST
// ESP32-S3 + INMP441
// 2nd-Order Butterworth Low-Pass Filter
// fs = 16000 Hz
// fc = 3000 Hz
// =====================================================

// ================= I2S CONFIG =================

#define I2S_SD      8
#define I2S_WS      6
#define I2S_SCK     7

#define I2S_PORT    I2S_NUM_0

#define SAMPLE_RATE 16000
#define BUFFER_LEN  256

// ================= FILTER COEFFICIENTS =================
// Generated using scipy.signal.butter(2, 3000, fs=16000)

const float b0 = 0.186694;
const float b1 = 0.373389;
const float b2 = 0.186694;

const float a1 = -0.462938;
const float a2 = 0.209715;

// ================= FILTER MEMORY =================

float x1 = 0.0;
float x2 = 0.0;

float y_prev1 = 0.0;
float y_prev2 = 0.0;

// ================= AUDIO BUFFERS =================

int32_t raw_samples[BUFFER_LEN];

int16_t raw16[BUFFER_LEN];

int16_t filtered16[BUFFER_LEN];

// =====================================================
// BUTTERWORTH FILTER FUNCTION
// =====================================================

float butterworthLPF(float x)
{
    float y =
        b0 * x +
        b1 * x1 +
        b2 * x2 -
        a1 * y_prev1 -
        a2 * y_prev2;

    // Update delay chain
    x2 = x1;
    x1 = x;

    y_prev2 = y_prev1;
    y_prev1 = y;

    return y;
}

// =====================================================
// SETUP
// =====================================================

void setup()
{
    Serial.begin(115200);

    delay(2000);

    Serial.println();
    Serial.println("=================================");
    Serial.println("AURA BAND DSP FILTER TEST");
    Serial.println("Butterworth LPF + INMP441");
    Serial.println("=================================");

    // ================= I2S CONFIG =================

    i2s_config_t i2s_config =
    {
        .mode =
            (i2s_mode_t)
            (I2S_MODE_MASTER | I2S_MODE_RX),

        .sample_rate = SAMPLE_RATE,

        .bits_per_sample =
            I2S_BITS_PER_SAMPLE_32BIT,

        .channel_format =
            I2S_CHANNEL_FMT_ONLY_LEFT,

        .communication_format =
            I2S_COMM_FORMAT_STAND_I2S,

        .intr_alloc_flags =
            ESP_INTR_FLAG_LEVEL1,

        .dma_buf_count = 8,

        .dma_buf_len = BUFFER_LEN,

        .use_apll = false
    };

    i2s_pin_config_t pin_config =
    {
        .bck_io_num = I2S_SCK,

        .ws_io_num = I2S_WS,

        .data_out_num =
            I2S_PIN_NO_CHANGE,

        .data_in_num = I2S_SD
    };

    i2s_driver_install(
        I2S_PORT,
        &i2s_config,
        0,
        NULL
    );

    i2s_set_pin(
        I2S_PORT,
        &pin_config
    );

    Serial.println("I2S READY");
}

// =====================================================
// LOOP
// =====================================================

void loop()
{
    size_t bytes_read = 0;

    // ==========================================
    // READ MICROPHONE DATA
    // ==========================================

    i2s_read(
        I2S_PORT,
        raw_samples,
        sizeof(raw_samples),
        &bytes_read,
        portMAX_DELAY
    );

    int sample_count = bytes_read / 4;

    // ==========================================
    // PROCESS AUDIO
    // ==========================================

    for(int i = 0; i < sample_count; i++)
    {
        // Convert 32-bit -> 16-bit
        raw16[i] = raw_samples[i] >> 13;

        // Apply Butterworth Filter
        float filtered =
            butterworthLPF(raw16[i]);

        // Store filtered output
        filtered16[i] =
            (int16_t)filtered;

        // ======================================
        // SEND TO PYTHON
        // raw,filtered
        // ======================================

        Serial.print(raw16[i]);

        Serial.print(",");

        Serial.println(filtered16[i]);
    }
}