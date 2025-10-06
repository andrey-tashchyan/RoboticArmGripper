/*
 * Strawberry Classifier (BGR input -> ripe / unripe decision)
 *
 * Compile:
 *   g++ -O3 -std=c++17 strawberry_classifier.cxx -o strawberry_classifier `pkg-config --cflags --libs opencv4` -D STRAWBERRY_DEMO
 *
 * Run demo:
 *   ./strawberry_classifier path/to/image.jpg
 *
 * This file exposes a C-compatible API (classify_strawberry_bgr) implemented in C++
 * with OpenCV. It analyses the central region of a frame and classifies whether
 * the dominant strawberry is red (ripe), white (unripe), or unknown.
 * The processing pipeline follows the 14 enumerated steps described in the task,
 * and every step is documented inline below.
 */

#include <opencv2/opencv.hpp>

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    STRAWBERRY_UNKNOWN = 0,
    STRAWBERRY_RED = 1,
    STRAWBERRY_WHITE = 2
} strawberry_label_t;

typedef struct {
    int red_h_low_1;
    int red_h_high_1;
    int red_h_low_2;
    int red_h_high_2;
    int red_s_min;
    int red_v_min;
    int white_s_max;
    int white_v_min;
    int glare_v_min;
    int glare_s_max;
    int green_h_low;
    int green_h_high;
    int green_s_min;
    int green_v_min;
    int roi_percent;
    int morph_open_kernel;
    int morph_close_kernel;
    float red_min_fraction;
    float white_min_fraction;
    float decision_margin;
    int v_min_considered;
    int downscale_factor;
    int enable_debug;
} strawberry_params_t;

typedef struct {
    strawberry_label_t label;
    float red_fraction;
    float white_fraction;
    float confidence;
    int considered_pixels;
} strawberry_result_t;

int classify_strawberry_bgr(const unsigned char* bgr_ptr,
                            int width,
                            int height,
                            int stride,
                            const strawberry_params_t* params,
                            strawberry_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

// Default configuration used when params == NULL.
static const strawberry_params_t kDefaultParams = {
    0,     /* red_h_low_1 */
    10,    /* red_h_high_1 */
    170,   /* red_h_low_2 */
    180,   /* red_h_high_2 */
    80,    /* red_s_min */
    80,    /* red_v_min */
    40,    /* white_s_max */
    160,   /* white_v_min */
    230,   /* glare_v_min */
    30,    /* glare_s_max */
    35,    /* green_h_low */
    85,    /* green_h_high */
    60,    /* green_s_min */
    60,    /* green_v_min */
    60,    /* roi_percent */
    3,     /* morph_open_kernel */
    5,     /* morph_close_kernel */
    0.25f, /* red_min_fraction */
    0.25f, /* white_min_fraction */
    0.10f, /* decision_margin */
    40,    /* v_min_considered */
    1,     /* downscale_factor */
    0      /* enable_debug */
};

// Utility helpers written in C style for reusability and clarity.
static inline int clamp_byte(int v) {
    if (v < 0) {
        return 0;
    }
    if (v > 255) {
        return 255;
    }
    return v;
}

static inline int clamp_hue(int v) {
    if (v < 0) {
        return 0;
    }
    if (v > 179) {
        return 179;
    }
    return v;
}

static inline int int_max(int a, int b) {
    return (a > b) ? a : b;
}

static inline int int_min(int a, int b) {
    return (a < b) ? a : b;
}

static void fill_unknown_result(strawberry_result_t* result) {
    if (result == NULL) {
        return;
    }
    result->label = STRAWBERRY_UNKNOWN;
    result->red_fraction = 0.0f;
    result->white_fraction = 0.0f;
    result->confidence = 0.0f;
    result->considered_pixels = 0;
}

extern "C" int classify_strawberry_bgr(const unsigned char* bgr_ptr,
                                       int width,
                                       int height,
                                       int stride,
                                       const strawberry_params_t* params,
                                       strawberry_result_t* out_result) {
    // Step 1: Validate inputs and fail fast on null pointers or invalid geometry.
    if (out_result == NULL) {
        return -1;
    }
    fill_unknown_result(out_result);
    if (bgr_ptr == NULL || width <= 0 || height <= 0 || stride < 0) {
        return -1;
    }

    // Step 13: Resolve configuration (use defaults when params == NULL).
    const strawberry_params_t* cfg = (params != NULL) ? params : &kDefaultParams;

    // Pre-clamp frequently used parameters to safe ranges.
    const int downscale_factor = int_max(1, cfg->downscale_factor);
    const int roi_percent = int_max(1, int_min(100, cfg->roi_percent));
    const int red_h_low_1 = clamp_hue(cfg->red_h_low_1);
    const int red_h_high_1 = clamp_hue(cfg->red_h_high_1);
    const int red_h_low_2 = clamp_hue(cfg->red_h_low_2);
    const int red_h_high_2 = clamp_hue(cfg->red_h_high_2);
    const int red_s_min = clamp_byte(cfg->red_s_min);
    const int red_v_min = clamp_byte(cfg->red_v_min);
    const int white_s_max = clamp_byte(cfg->white_s_max);
    const int white_v_min = clamp_byte(cfg->white_v_min);
    const int glare_v_min = clamp_byte(cfg->glare_v_min);
    const int glare_s_max = clamp_byte(cfg->glare_s_max);
    const int green_h_low = clamp_hue(cfg->green_h_low);
    const int green_h_high = clamp_hue(cfg->green_h_high);
    const int green_s_min = clamp_byte(cfg->green_s_min);
    const int green_v_min = clamp_byte(cfg->green_v_min);
    const int v_min_considered = clamp_byte(cfg->v_min_considered);
    const int morph_open_kernel = int_max(1, cfg->morph_open_kernel);
    const int morph_close_kernel = int_max(1, cfg->morph_close_kernel);

    // Step 2: Wrap raw BGR pointer in cv::Mat, verifying stride.
    const size_t required_stride = (size_t)width * 3u;
    const size_t actual_stride = (stride == 0) ? required_stride : (size_t)stride;
    if (actual_stride < required_stride) {
        return -1;
    }
    cv::Mat bgr(height, width, CV_8UC3, (void*)bgr_ptr, actual_stride);

    cv::Mat bgr_view;
    if (downscale_factor > 1) {
        // Downscale to suppress noise and accelerate subsequent operations.
        const double scale = 1.0 / (double)downscale_factor;
        cv::resize(bgr, bgr_view, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        bgr_view = bgr;
    }

    // Step 3: Select centered ROI covering roi_percent% of both dimensions.
    const double roi_ratio = (double)roi_percent / 100.0;
    int roi_width = (int)((double)bgr_view.cols * roi_ratio + 0.5);
    int roi_height = (int)((double)bgr_view.rows * roi_ratio + 0.5);
    roi_width = int_max(1, int_min(bgr_view.cols, roi_width));
    roi_height = int_max(1, int_min(bgr_view.rows, roi_height));
    const int roi_x = (bgr_view.cols - roi_width) / 2;
    const int roi_y = (bgr_view.rows - roi_height) / 2;
    const cv::Rect roi_rect(roi_x, roi_y, roi_width, roi_height);
    cv::Mat bgr_roi = bgr_view(roi_rect);

    // Step 4: Convert ROI to HSV color space for robust color segmentation.
    cv::Mat hsv_roi;
    cv::cvtColor(bgr_roi, hsv_roi, cv::COLOR_BGR2HSV);

    // Split channels to simplify component-wise thresholding operations.
    cv::Mat hsv_channels[3];
    cv::split(hsv_roi, hsv_channels);
    const cv::Mat& hue = hsv_channels[0];
    const cv::Mat& sat = hsv_channels[1];
    const cv::Mat& val = hsv_channels[2];

    // Step 5a: Create mask for glare (very bright, low saturation).
    cv::Mat mask_glare;
    {
        cv::Mat mask_v_high, mask_s_low;
        cv::inRange(val, glare_v_min, 255, mask_v_high);
        cv::inRange(sat, 0, glare_s_max, mask_s_low);
        cv::bitwise_and(mask_v_high, mask_s_low, mask_glare);
    }

    // Step 5b: Create mask for green regions (likely leaves/stems) to exclude.
    cv::Mat mask_green;
    {
        cv::Mat mask_h_green, mask_s_green, mask_v_green, temp;
        if (green_h_low <= green_h_high) {
            cv::inRange(hue, green_h_low, green_h_high, mask_h_green);
        } else {
            // Handle wrap-around hue ranges (e.g., [170,179] U [0,10]).
            cv::Mat range1, range2;
            cv::inRange(hue, green_h_low, 179, range1);
            cv::inRange(hue, 0, green_h_high, range2);
            cv::bitwise_or(range1, range2, mask_h_green);
        }
        cv::inRange(sat, green_s_min, 255, mask_s_green);
        cv::inRange(val, green_v_min, 255, mask_v_green);
        cv::bitwise_and(mask_h_green, mask_s_green, temp);
        cv::bitwise_and(temp, mask_v_green, mask_green);
    }

    // Step 5c: Keep pixels that are bright enough and not glare/green.
    cv::Mat mask_considered;
    {
        cv::Mat mask_v_valid, mask_not_glare, mask_not_green, temp;
        cv::inRange(val, v_min_considered, 255, mask_v_valid);
        cv::bitwise_not(mask_glare, mask_not_glare);
        cv::bitwise_not(mask_green, mask_not_green);
        cv::bitwise_and(mask_v_valid, mask_not_glare, temp);
        cv::bitwise_and(temp, mask_not_green, mask_considered);
    }

    // Step 6: Build red mask from two hue intervals intersected with considered pixels.
    cv::Mat mask_red_combined = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);
    if (red_h_low_1 <= red_h_high_1) {
        cv::Mat mask_red_1;
        cv::inRange(hsv_roi,
                    cv::Scalar(red_h_low_1, red_s_min, red_v_min),
                    cv::Scalar(red_h_high_1, 255, 255),
                    mask_red_1);
        cv::bitwise_or(mask_red_combined, mask_red_1, mask_red_combined);
    }
    if (red_h_low_2 <= red_h_high_2) {
        cv::Mat mask_red_2;
        cv::inRange(hsv_roi,
                    cv::Scalar(red_h_low_2, red_s_min, red_v_min),
                    cv::Scalar(red_h_high_2, 255, 255),
                    mask_red_2);
        cv::bitwise_or(mask_red_combined, mask_red_2, mask_red_combined);
    }
    cv::Mat mask_red;
    cv::bitwise_and(mask_red_combined, mask_considered, mask_red);

    // Step 7: Build white mask (low saturation, sufficient brightness) within considered pixels.
    cv::Mat mask_white;
    {
        cv::Mat mask_s_low, mask_v_high, mask_white_raw;
        cv::inRange(sat, 0, white_s_max, mask_s_low);
        cv::inRange(val, white_v_min, 255, mask_v_high);
        cv::bitwise_and(mask_s_low, mask_v_high, mask_white_raw);
        cv::bitwise_and(mask_white_raw, mask_considered, mask_white);
    }

    // Step 8: Clean masks with optional morphology (open/close) when kernels > 1.
    if (morph_open_kernel > 1) {
        cv::Mat open_kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(morph_open_kernel, morph_open_kernel));
        cv::morphologyEx(mask_red, mask_red, cv::MORPH_OPEN, open_kernel);
        cv::morphologyEx(mask_white, mask_white, cv::MORPH_OPEN, open_kernel);
    }
    if (morph_close_kernel > 1) {
        cv::Mat close_kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(morph_close_kernel, morph_close_kernel));
        cv::morphologyEx(mask_red, mask_red, cv::MORPH_CLOSE, close_kernel);
        cv::morphologyEx(mask_white, mask_white, cv::MORPH_CLOSE, close_kernel);
    }

    // Step 9: Count pixels for statistics and handle the empty case.
    const int N_considered = cv::countNonZero(mask_considered);
    const int N_red = cv::countNonZero(mask_red);
    const int N_white = cv::countNonZero(mask_white);
    if (N_considered == 0) {
        out_result->considered_pixels = 0;
        out_result->label = STRAWBERRY_UNKNOWN;
        out_result->red_fraction = 0.0f;
        out_result->white_fraction = 0.0f;
        out_result->confidence = 0.0f;
        return 0;
    }

    // Step 10: Compute fractions and confidence (absolute difference).
    const float red_fraction = (float)N_red / (float)N_considered;
    const float white_fraction = (float)N_white / (float)N_considered;
    const float confidence = (float)fabsf(red_fraction - white_fraction);

    // Step 11: Apply decision logic using configured thresholds and margin.
    strawberry_label_t label = STRAWBERRY_UNKNOWN;
    if (red_fraction >= cfg->red_min_fraction &&
        red_fraction >= white_fraction + cfg->decision_margin) {
        label = STRAWBERRY_RED;
    } else if (white_fraction >= cfg->white_min_fraction &&
               white_fraction >= red_fraction + cfg->decision_margin) {
        label = STRAWBERRY_WHITE;
    }

    // Step 12: Populate result structure for the caller.
    out_result->label = label;
    out_result->red_fraction = red_fraction;
    out_result->white_fraction = white_fraction;
    out_result->confidence = confidence;
    out_result->considered_pixels = N_considered;

    // Step 14: Emit debug diagnostics when requested.
    if (cfg->enable_debug != 0) {
        fprintf(stderr,
                "[strawberry_classifier] input=%dx%d stride=%llu, downscale=%d, "
                "ROI=%dx%d at (%d,%d)\n",
                width,
                height,
                (unsigned long long)actual_stride,
                downscale_factor,
                roi_width,
                roi_height,
                roi_x,
                roi_y);
        fprintf(stderr,
                "[strawberry_classifier] considered=%d red=%d white=%d "
                "(red_fraction=%.3f white_fraction=%.3f confidence=%.3f) -> label=%d\n",
                N_considered,
                N_red,
                N_white,
                red_fraction,
                white_fraction,
                confidence,
                (int)label);
    }

    return 0;
}

#ifdef STRAWBERRY_DEMO
// Demo executable: load an image, run classifier, print decision, and display ROI.
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s image_path\n", argv[0]);
        return 1;
    }

    // Load a BGR image; OpenCV returns an empty matrix on failure.
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        fprintf(stderr, "Failed to load image: %s\n", argv[1]);
        return 1;
    }

    strawberry_result_t result;
    if (classify_strawberry_bgr(image.data,
                                image.cols,
                                image.rows,
                                0,
                                NULL,
                                &result) != 0) {
        fprintf(stderr, "Classification failed.\n");
        return 1;
    }

    const char* label_str = "UNKNOWN";
    if (result.label == STRAWBERRY_RED) {
        label_str = "RED";
    } else if (result.label == STRAWBERRY_WHITE) {
        label_str = "WHITE";
    }
    printf("Decision: %s, red_fraction=%.3f, white_fraction=%.3f, confidence=%.3f\n",
           label_str,
           result.red_fraction,
           result.white_fraction,
           result.confidence);

    // Visualise the ROI so users can confirm the analysed area.
    const int roi_percent = int_max(1, int_min(100, kDefaultParams.roi_percent));
    const double roi_ratio = (double)roi_percent / 100.0;
    int roi_width = (int)((double)image.cols * roi_ratio + 0.5);
    int roi_height = (int)((double)image.rows * roi_ratio + 0.5);
    roi_width = int_max(1, int_min(image.cols, roi_width));
    roi_height = int_max(1, int_min(image.rows, roi_height));
    const int roi_x = (image.cols - roi_width) / 2;
    const int roi_y = (image.rows - roi_height) / 2;
    cv::rectangle(image, cv::Rect(roi_x, roi_y, roi_width, roi_height), cv::Scalar(0, 255, 255), 2);
    cv::imshow("Strawberry ROI", image);
    cv::waitKey(0);
    return 0;
}
#endif  // STRAWBERRY_DEMO
