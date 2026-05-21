#include <Arduino.h>
#include <driver/i2s.h>
#include <Adafruit_NeoPixel.h>

/* * =========================================================================
 * BƯỚC QUAN TRỌNG: INCLUDE THƯ VIỆN AI CỦA THẢO
 * Chú ý: Tên file này phải trùng với tên thư viện xuất ra từ Edge Impulse.
 * (Dựa theo platformio.ini của Thảo, nó tên là Final_Project_inferencing.h)
 * =========================================================================
 */
#include <Final_Project_inferencing.h>

// ============ HARDWARE CONFIGURATION (STABLE) ============
#define I2S_SD      8   // D5 (Microphone Data)
#define I2S_WS      6   // D3 (LRCLK/WS)
#define I2S_SCK     7   // D4 (BCLK/SCK)
#define I2S_PORT    I2S_NUM_0
 
#define LED_PIN     21  // D10
#define NUM_LEDS    16
#define BUZZER_PIN  17  // D8 (Active Buzzer)

Adafruit_NeoPixel ring = Adafruit_NeoPixel(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

// ============ AUDIO CONFIGURATION ============
#define AUDIO_SAMPLE_RATE 16000
#define I2S_BUFFER_LEN    256  // Samples per read from Mic DMA

QueueHandle_t audioQueue;

// Buffer for 1 second of audio (16000 samples) for Edge Impulse
int16_t ei_audio_buffer[EI_CLASSIFIER_RAW_SAMPLE_COUNT];
int ei_buffer_index = 0;

// ============ ADAPTIVE CONFIDENCE THRESHOLDS ============
// Different thresholds for different commands
struct ConfidenceThresholds {
    float aura_wake = 0.70f;
    float walk = 0.50f;
    float cross = 0.50f;
    float help = 0.50f;
    float cancel = 0.50f;
    float background_reject = 0.65f;
};

ConfidenceThresholds thresholds;

// ============ STATE MACHINE ============
enum DeviceState { SLEEP, AURA_WAKE, WALK, CROSS, HELP };
volatile DeviceState currentState = SLEEP;
 
// ============ TIMERS ============
unsigned long bootTime = 0;
unsigned long lastInferenceTime = 0;
unsigned long wakeTime = 0;
 
// ============ FLAGS ============
bool isAwake = false;

// ============ INFERENCE DEBOUNCING ============
#define INFERENCE_COOLDOWN_MS 300      // Minimum time between inferences
#define WAKE_TIMEOUT_MS 10000          // How long to stay awake after last command
#define BOOT_SAFE_PERIOD_MS 3000       // Ignore detections during boot
#define COMMAND_VALIDATION_WINDOW 2000 // Time window to check for consistent detections

// =======================================================
// HÀM CALLBACK CHO EDGE IMPULSE LẤY DATA
// =======================================================
int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    numpy::int16_to_float(&ei_audio_buffer[offset], out_ptr, length);
    return 0;
}

// ============ AUDIO AMPLITUDE DETECTION ============
float calculateAudioAmplitude(int16_t* samples, int count) {
    int32_t sum = 0;
    for (int i = 0; i < count; i++) {
        sum += abs(samples[i]);
    }
    return (float)sum / count;
}

// ============ VALIDATE COMMAND WITH MULTI-FRAME CHECK ============
bool validateCommandConsistency(String command, float confidence) {
    // Additional validation: Check if we're getting consistent detections
    static String lastValidCommand = "";
    static unsigned long lastValidCommandTime = 0;
    static int consistencyCounter = 0;
    
    if (command == lastValidCommand && 
        millis() - lastValidCommandTime < COMMAND_VALIDATION_WINDOW) {
        consistencyCounter++;
    } else {
        consistencyCounter = 1;
        lastValidCommand = command;
    }
    
    lastValidCommandTime = millis();
    
    // For WALK, CROSS, HELP: require 1 strong detection
    // For AURA: require 2 consecutive validations for extra safety
    if (command == "AURA") {
        return consistencyCounter >= 2 && confidence > thresholds.aura_wake;
    }
    
    return consistencyCounter >= 1;
}

// =======================================================
// IIR LOW PASS FILTER
// =======================================================

float alpha = 0.5;
float previous_output = 0;

int16_t applyIIRFilter(int16_t input)
{
    float output = alpha * input +
                   (1.0 - alpha) * previous_output;

    previous_output = output;

    return (int16_t)output;
}

// =======================================================
// TASK 1: THU ÂM CHẠY TRÊN CORE 0 (THE EARS)
// =======================================================
void AudioCaptureTask(void *parameter) {
    // 1. Configure I2S DMA for automatic mic reading
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = AUDIO_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = I2S_BUFFER_LEN,
        .use_apll = false
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK, 
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE, 
        .data_in_num = I2S_SD
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);

    Serial.println("[Core 0] Audio Capture Task Started!");

    int32_t raw_samples[I2S_BUFFER_LEN];
    int16_t processed_samples[I2S_BUFFER_LEN];
    size_t bytes_read = 0;
    unsigned long lastPrint = 0;

    while(1) {
        // Read from microphone (blocks until DMA buffer is full)
        i2s_read(I2S_PORT, &raw_samples, sizeof(raw_samples), &bytes_read, portMAX_DELAY);
        
        if (bytes_read > 0) {
            int num_samples = bytes_read / 4; // int32_t = 4 bytes
            
            // Extract 16-bit samples and reduce noise
            for (int i = 0; i < num_samples; i++) {
                int16_t raw16 = raw_samples[i] >> 13; // Remove noise bits

                int16_t filtered = applyIIRFilter(raw16);

                processed_samples[i] = filtered;

                // Save ONLY sample pairs for Python plotting
                //Serial.printf("%d,%d\n", raw16, filtered);
            }

            // Debug mic level
            if (millis() - lastPrint > 1000) {

                float level =
                    calculateAudioAmplitude(
                        processed_samples,
                        num_samples);

                Serial.print("MIC LEVEL: ");
                Serial.println(level);

                lastPrint = millis();
            }
            
            // Send to Core 1 for processing
            xQueueSend(audioQueue, processed_samples, 0);
        }
    }
}

// =======================================================
// TASK 2: AI INFERENCE CHẠY TRÊN CORE 1 (THE BRAIN)
// =======================================================
void InferenceTask(void *parameter) {
    Serial.println("[Core 1] AI Inference Task Started!");
    
    int16_t received_audio[I2S_BUFFER_LEN];
 
    while(1) {
        // Receive audio from Core 0
        if (xQueueReceive(audioQueue, &received_audio, portMAX_DELAY) == pdPASS) {
            
            // Calculate audio amplitude for voice activity detection
            float amplitude = calculateAudioAmplitude(received_audio, I2S_BUFFER_LEN);
            
            // Fill the AI buffer
            for (int i = 0; i < I2S_BUFFER_LEN; i++) {
                if(ei_buffer_index < EI_CLASSIFIER_RAW_SAMPLE_COUNT) {
                    ei_audio_buffer[ei_buffer_index++] = received_audio[i];
                }
                
                // When we have 1 second of audio -> Run inference
                if (ei_buffer_index >= EI_CLASSIFIER_RAW_SAMPLE_COUNT) {
                    ei_buffer_index = 0;
                    
                    // ============ INFERENCE COOLDOWN CHECK ============
                    if (millis() - lastInferenceTime < INFERENCE_COOLDOWN_MS) { 
                        continue;
                    }
                    
                    // ============ BOOT SAFETY PERIOD ============
                    if (millis() - bootTime < BOOT_SAFE_PERIOD_MS) {
                        continue;
                    }
 
                    lastInferenceTime = millis();
 
                    // Run the Edge Impulse model
                    signal_t signal;
                    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
                    signal.get_data = &microphone_audio_signal_get_data;
                    ei_impulse_result_t result = { 0 };
 
                    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, false);
                    if (r != EI_IMPULSE_OK) {
                        // Serial.printf("Inference error: %d\n", r);
                        continue;
                    }
 
                    // Find best prediction
                    int best_idx = 0;
                    float best_score = 0.0;
                    for (uint16_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
                        if (result.classification[ix].value > best_score) {
                            best_score = result.classification[ix].value;
                            best_idx = ix;
                        }
                    }
 
                    String predicted_word = result.classification[best_idx].label;

                    // ======================================
                    // DEBUG
                    // ======================================

                    Serial.println("");
                    Serial.println("========== RESULT ==========");

                    Serial.print("BEST: ");
                    Serial.print(predicted_word);

                    Serial.print(" | ");

                    Serial.println(best_score, 3);

                    for (uint16_t ix = 0;
                        ix < EI_CLASSIFIER_LABEL_COUNT;
                        ix++) {

                        Serial.print(
                            result.classification[ix].label);

                        Serial.print(": ");

                        Serial.println(
                            result.classification[ix].value,
                            3);
                    }

                    Serial.println("============================");
                    Serial.println("");

                    // ======================================
                    // FILTER LOW SCORE
                    // ======================================

                    if (best_score <
                        thresholds.background_reject) {

                        continue;
                    }

                    // ======================================
                    // WAKE MODE
                    // ======================================

                    if (!isAwake) {

                        if (predicted_word == "AURA" &&
                            best_score > thresholds.aura_wake) {

                            isAwake = true;

                            wakeTime = millis();

                            currentState = AURA_WAKE;

                            Serial.println("");
                            Serial.println("WAKE WORD: AURA");
                            Serial.println("");
                        }

                        continue;
                    }

                    // ======================================
                    // TIMEOUT
                    // ======================================

                    if (millis() - wakeTime >
                        WAKE_TIMEOUT_MS) {

                        isAwake = false;

                        currentState = SLEEP;

                        Serial.println("Sleep Mode");

                        continue;
                    }

                    // refresh timeout
                    wakeTime = millis();

                    // ======================================
                    // COMMANDS
                    // ======================================

                    if (predicted_word == "WALK" &&
                        best_score > thresholds.walk) {

                        currentState = WALK;

                        Serial.println("COMMAND: WALK");
                    }

                    else if (predicted_word == "CROSS" &&
                             best_score > thresholds.cross) {

                        currentState = CROSS;

                        Serial.println("COMMAND: CROSS");
                    }

                    else if (predicted_word == "HELP" &&
                             best_score > thresholds.help) {

                        currentState = HELP;

                        Serial.println("COMMAND: HELP");
                    }

                    else if (predicted_word == "CANCEL" &&
                             best_score > thresholds.cancel) {

                        currentState = SLEEP;

                        isAwake = false;

                        Serial.println("COMMAND: CANCEL");
                    }
                }
            }
        }
    }
}

// =======================================================
// TASK 3: OUTPUT ANIMATION CHẠY TRÊN CORE 1 (THE HANDS/MOUTH)
// Đọc State Machine và chớp đèn/kêu còi độc lập
// =======================================================
void OutputTask(void *parameter) {
    int aura_idx = 0;
    bool toggle = false;
    unsigned long animationStartTime = 0;
 
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);
 
    Serial.println("[Core 1] Output Animation Task Started!");
 
    while(1) {
 
        switch(currentState) {
            case SLEEP: 
                    ring.clear();
                    ring.setPixelColor(0, ring.Color(255, 0, 0)); // Single red LED
                    ring.show();
                    digitalWrite(BUZZER_PIN, LOW);
                    vTaskDelay(pdMS_TO_TICKS(200));
                break;
 
            case AURA_WAKE: 
                    // Rotating cyan animation
                    ring.clear();
                    ring.setPixelColor(aura_idx, ring.Color(0, 255, 255));
                    ring.setPixelColor((aura_idx + 1) % NUM_LEDS, ring.Color(0, 200, 200)); // Trailing glow
                    ring.show();
                    aura_idx = (aura_idx + 1) % NUM_LEDS;
                    digitalWrite(BUZZER_PIN, LOW);
                    vTaskDelay(pdMS_TO_TICKS(40)); // Smooth rotation
                break;
 
            case WALK: 
                    // Static yellow all LEDs
                    for(int i = 0; i < NUM_LEDS; i++) {
                        ring.setPixelColor(i, ring.Color(255, 200, 0));
                    }
                    ring.show();
                    digitalWrite(BUZZER_PIN, LOW);
                    vTaskDelay(pdMS_TO_TICKS(100));
                break;
 
            case CROSS:
                    // Flashing orange with buzzer
                    toggle = !toggle;
                    if(toggle) {
                        for(int i = 0; i < NUM_LEDS; i++) {
                            ring.setPixelColor(i, ring.Color(255, 100, 0)); // Orange
                        }
                        ring.show();
                        digitalWrite(BUZZER_PIN, HIGH);
                    } else {
                        ring.clear();
                        ring.show();
                        digitalWrite(BUZZER_PIN, LOW);
                    }
                    vTaskDelay(pdMS_TO_TICKS(300)); // 600ms period
                break;
 
            case HELP: 
                    // Rapid red flashing with loud buzzer
                    toggle = !toggle;
                    if(toggle) {
                        for(int i = 0; i < NUM_LEDS; i++) {
                            ring.setPixelColor(i, ring.Color(255, 0, 0)); // Bright red
                        }
                        ring.show();
                        digitalWrite(BUZZER_PIN, HIGH);
                    } else {
                        ring.clear();
                        ring.show();
                        digitalWrite(BUZZER_PIN, LOW);
                    }
                    vTaskDelay(pdMS_TO_TICKS(100)); // 200ms period - urgent!
                break;
        }
    }
}

// ============ MAIN SETUP ============
void setup() {
    bootTime = millis();
 
    // Mute buzzer immediately
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW); 
 
    Serial.begin(115200);

    delay(2000);

    Serial.println("");
    Serial.println("AURA BAND STARTING...");
    Serial.println("");
 
    // Initialize NeoPixel ring
    ring.begin();
    ring.setBrightness(40);
    for(int i = 0; i < NUM_LEDS; i++) {
        ring.setPixelColor(i, ring.Color(0, 100, 255)); // Boot blue
    }
    ring.show();
    delay(500);
    ring.clear();
    ring.show();
 
    // Create queue for audio data transfer (20 slots, each 256 int16_t samples)
    audioQueue = xQueueCreate(20, sizeof(int16_t) * I2S_BUFFER_LEN); 
 
    // TASK 1: Audio Capture (Core 0, Priority 3 - Highest)
    xTaskCreatePinnedToCore(
        AudioCaptureTask,
        "AudioTask",
        10240,
        NULL,
        3,
        NULL,
        0
    );
    
    // TASK 2: AI Inference (Core 1, Priority 2 - Medium)
    xTaskCreatePinnedToCore(
        InferenceTask,
        "InferenceTask",
        16384,  // Large stack for AI operations
        NULL,
        2,
        NULL,
        1
    );
    
    // TASK 3: Output Animation (Core 1, Priority 1 - Low)
    xTaskCreatePinnedToCore(
        OutputTask,
        "OutputTask",
        4096,
        NULL,
        1,
        NULL,
        1
    );
 
    // Flash ready indicator
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < NUM_LEDS; j++) {
            ring.setPixelColor(j, ring.Color(0, 255, 0));
        }
        ring.show();
        delay(150);
        ring.clear();
        ring.show();
        delay(150);
    }
}
 
void loop() {
    // FreeRTOS handles everything
    vTaskDelay(pdMS_TO_TICKS(1000));
}