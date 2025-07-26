#ifndef CURVEAPPEARANCE_H
#define CURVEAPPEARANCE_H

#include <QColor>
#include <QMetaType>

namespace BC::Data {

// Curve appearance storage keys
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

/**
 * @brief Pure data enums for curve visual appearance properties
 * 
 * These enums mirror the QWT and Qt enums but are independent of GUI libraries.
 * Conversion functions can map between these and the actual GUI enums.
 */

enum class CurveStyle {
    NoCurve,
    Lines,
    Sticks,
    Steps,
    Dots
};

enum class LineStyle {
    NoPen,
    SolidLine,
    DashLine,
    DotLine,
    DashDotLine,
    DashDotDotLine
};

enum class MarkerStyle {
    NoSymbol,
    Ellipse,
    Rect,
    Diamond,
    Triangle,
    DTriangle,
    UTriangle,
    LTriangle,
    RTriangle,
    Cross,
    XCross,
    HLine,
    VLine,
    Star1,
    Star2,
    Hexagon
};

enum class YAxis {
    Left,
    Right
};

/**
 * @brief Pure data structure for curve visual appearance properties
 * 
 * This structure contains all the visual properties needed to render a curve
 * without any GUI widget dependencies. It can be serialized/deserialized
 * and converted to/from GUI widget types as needed.
 */
struct CurveAppearance {
    // Visual properties
    QColor color{Qt::blue};
    CurveStyle curveStyle{CurveStyle::Lines};
    double lineThickness{1.0};
    LineStyle lineStyle{LineStyle::SolidLine};
    MarkerStyle markerStyle{MarkerStyle::NoSymbol};
    int markerSize{6};
    bool visible{true};
    bool autoscale{true};
    YAxis yAxis{YAxis::Left};
    
    // Default constructor
    CurveAppearance() = default;
    
    // Equality comparison
    bool operator==(const CurveAppearance &other) const {
        return color == other.color &&
               curveStyle == other.curveStyle &&
               qFuzzyCompare(lineThickness, other.lineThickness) &&
               lineStyle == other.lineStyle &&
               markerStyle == other.markerStyle &&
               markerSize == other.markerSize &&
               visible == other.visible &&
               autoscale == other.autoscale &&
               yAxis == other.yAxis;
    }
    
    bool operator!=(const CurveAppearance &other) const {
        return !(*this == other);
    }
};

} // namespace BC::Data

// Register for Qt's meta type system
Q_DECLARE_METATYPE(BC::Data::CurveAppearance)
Q_DECLARE_METATYPE(BC::Data::CurveStyle)
Q_DECLARE_METATYPE(BC::Data::LineStyle)
Q_DECLARE_METATYPE(BC::Data::MarkerStyle)
Q_DECLARE_METATYPE(BC::Data::YAxis)

#endif // CURVEAPPEARANCE_H