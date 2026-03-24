#ifndef CURVEAPPEARANCE_H
#define CURVEAPPEARANCE_H

#include <QString>

namespace BC::Data {

// Curve appearance storage keys used by overlay metadata
namespace CurveKey {
    static const QString curve{"Curve"};
    static const QString color{"color"};
    static const QString curveStyle{"curveStyle"};
    static const QString lineStyle{"lineStyle"};
    static const QString thickness{"thickness"};
    static const QString marker{"marker"};
    static const QString markerSize{"markerSize"};
    static const QString axisX{"xAxis"};
    static const QString axisY{"yAxis"};
    static const QString visible{"visible"};
    static const QString autoscale{"autoscale"};
    static const QString plotIndex{"plotIndex"};
}

} // namespace BC::Data

#endif // CURVEAPPEARANCE_H
