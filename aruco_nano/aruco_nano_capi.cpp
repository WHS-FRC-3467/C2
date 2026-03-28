#include "aruco_nano.h"
#include <cstring>

extern "C" {

struct ArucoNanoDetector {
    aruco_nano::Params params;
};

struct Detection {
    int id;
    float corners[8]; // x0,y0, x1,y1, x2,y2, x3,y3
};

ArucoNanoDetector* aruco_nano_create(int dict_id) {
    auto* det = new ArucoNanoDetector();
    det->params.dict = cv::aruco::getPredefinedDictionary(dict_id);
    return det;
}

void aruco_nano_destroy(ArucoNanoDetector* det) {
    delete det;
}

int aruco_nano_detect(ArucoNanoDetector* det,
                      const uint8_t* data, int width, int height, int stride,
                      Detection* out, int max_detections) {
    cv::Mat image(height, width, CV_8UC1, const_cast<uint8_t*>(data), stride);
    auto markers = aruco_nano::MarkerDetector::detect(image, det->params);

    int count = 0;
    for (const auto& m : markers) {
        if (count >= max_detections) break;
        if (m.size() != 4) continue;
        out[count].id = m.id;
        for (int j = 0; j < 4; j++) {
            out[count].corners[j * 2]     = m[j].x;
            out[count].corners[j * 2 + 1] = m[j].y;
        }
        count++;
    }
    return count;
}

} // extern "C"
