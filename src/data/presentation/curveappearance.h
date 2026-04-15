#ifndef CURVEAPPEARANCE_H
#define CURVEAPPEARANCE_H

#include <QString>
#include <QLatin1StringView>

namespace BC::Data {

// Curve appearance storage keys used by overlay metadata
namespace CurveKey {
    inline constexpr QLatin1StringView curve{"Curve"};
    inline constexpr QLatin1StringView color{"color"};
    inline constexpr QLatin1StringView curveStyle{"curveStyle"};
    inline constexpr QLatin1StringView lineStyle{"lineStyle"};
    inline constexpr QLatin1StringView thickness{"thickness"};
    inline constexpr QLatin1StringView marker{"marker"};
    inline constexpr QLatin1StringView markerSize{"markerSize"};
    inline constexpr QLatin1StringView axisX{"xAxis"};
    inline constexpr QLatin1StringView axisY{"yAxis"};
    inline constexpr QLatin1StringView visible{"visible"};
    inline constexpr QLatin1StringView autoscale{"autoscale"};
    inline constexpr QLatin1StringView plotIndex{"plotIndex"};
}

} // namespace BC::Data

#endif // CURVEAPPEARANCE_H
