#include "curveappearance.h"

#include <QMetaEnum>
#include <QDebug>

namespace BC::Data {

// Register meta types for Qt's type system
namespace {
    // Register meta types at static initialization time
    struct MetaTypeRegistrar {
        MetaTypeRegistrar() {
            qRegisterMetaType<CurveAppearance>();
            qRegisterMetaType<CurveStyle>();
            qRegisterMetaType<LineStyle>();
            qRegisterMetaType<MarkerStyle>();
            qRegisterMetaType<YAxis>();
        }
    };
    static MetaTypeRegistrar registrar;
}

} // namespace BC::Data

// Debug stream operators for easier debugging
QDebug operator<<(QDebug debug, BC::Data::CurveStyle style) {
    QDebugStateSaver saver(debug);
    switch (style) {
    case BC::Data::CurveStyle::NoCurve: debug.nospace() << "NoCurve"; break;
    case BC::Data::CurveStyle::Lines: debug.nospace() << "Lines"; break;
    case BC::Data::CurveStyle::Sticks: debug.nospace() << "Sticks"; break;
    case BC::Data::CurveStyle::Steps: debug.nospace() << "Steps"; break;
    case BC::Data::CurveStyle::Dots: debug.nospace() << "Dots"; break;
    }
    return debug;
}

QDebug operator<<(QDebug debug, BC::Data::LineStyle style) {
    QDebugStateSaver saver(debug);
    switch (style) {
    case BC::Data::LineStyle::NoPen: debug.nospace() << "NoPen"; break;
    case BC::Data::LineStyle::SolidLine: debug.nospace() << "SolidLine"; break;
    case BC::Data::LineStyle::DashLine: debug.nospace() << "DashLine"; break;
    case BC::Data::LineStyle::DotLine: debug.nospace() << "DotLine"; break;
    case BC::Data::LineStyle::DashDotLine: debug.nospace() << "DashDotLine"; break;
    case BC::Data::LineStyle::DashDotDotLine: debug.nospace() << "DashDotDotLine"; break;
    }
    return debug;
}

QDebug operator<<(QDebug debug, BC::Data::MarkerStyle style) {
    QDebugStateSaver saver(debug);
    switch (style) {
    case BC::Data::MarkerStyle::NoSymbol: debug.nospace() << "NoSymbol"; break;
    case BC::Data::MarkerStyle::Ellipse: debug.nospace() << "Ellipse"; break;
    case BC::Data::MarkerStyle::Rect: debug.nospace() << "Rect"; break;
    case BC::Data::MarkerStyle::Diamond: debug.nospace() << "Diamond"; break;
    case BC::Data::MarkerStyle::Triangle: debug.nospace() << "Triangle"; break;
    case BC::Data::MarkerStyle::DTriangle: debug.nospace() << "DTriangle"; break;
    case BC::Data::MarkerStyle::UTriangle: debug.nospace() << "UTriangle"; break;
    case BC::Data::MarkerStyle::LTriangle: debug.nospace() << "LTriangle"; break;
    case BC::Data::MarkerStyle::RTriangle: debug.nospace() << "RTriangle"; break;
    case BC::Data::MarkerStyle::Cross: debug.nospace() << "Cross"; break;
    case BC::Data::MarkerStyle::XCross: debug.nospace() << "XCross"; break;
    case BC::Data::MarkerStyle::HLine: debug.nospace() << "HLine"; break;
    case BC::Data::MarkerStyle::VLine: debug.nospace() << "VLine"; break;
    case BC::Data::MarkerStyle::Star1: debug.nospace() << "Star1"; break;
    case BC::Data::MarkerStyle::Star2: debug.nospace() << "Star2"; break;
    case BC::Data::MarkerStyle::Hexagon: debug.nospace() << "Hexagon"; break;
    }
    return debug;
}

QDebug operator<<(QDebug debug, BC::Data::YAxis axis) {
    QDebugStateSaver saver(debug);
    switch (axis) {
    case BC::Data::YAxis::Left: debug.nospace() << "Left"; break;
    case BC::Data::YAxis::Right: debug.nospace() << "Right"; break;
    }
    return debug;
}

QDebug operator<<(QDebug debug, const BC::Data::CurveAppearance &appearance) {
    QDebugStateSaver saver(debug);
    debug.nospace() << "CurveAppearance("
                    << "color=" << appearance.color.name()
                    << ", style=" << appearance.curveStyle
                    << ", thickness=" << appearance.lineThickness
                    << ", lineStyle=" << appearance.lineStyle
                    << ", marker=" << appearance.markerStyle
                    << ", markerSize=" << appearance.markerSize
                    << ", visible=" << appearance.visible
                    << ", autoscale=" << appearance.autoscale
                    << ", yAxis=" << appearance.yAxis
                    << ")";
    return debug;
}