#include "curveappearanceconverter.h"
#include <qwt6/qwt_plot.h>

// =============================================================================
// Main conversion functions
// =============================================================================

CurveAppearanceWidget::CurveAppearance CurveAppearanceConverter::toGuiAppearance(const BC::Data::CurveAppearance &dataAppearance)
{
    CurveAppearanceWidget::CurveAppearance gui;
    
    gui.color = dataAppearance.color;
    gui.curveStyle = toQwtCurveStyle(dataAppearance.curveStyle);
    gui.lineThickness = dataAppearance.lineThickness;
    gui.lineStyle = toQtPenStyle(dataAppearance.lineStyle);
    gui.markerStyle = toQwtSymbolStyle(dataAppearance.markerStyle);
    gui.markerSize = dataAppearance.markerSize;
    gui.visible = dataAppearance.visible;
    gui.autoscale = dataAppearance.autoscale;
    gui.yAxis = toQwtAxisId(dataAppearance.yAxis);
    
    return gui;
}

BC::Data::CurveAppearance CurveAppearanceConverter::toDataAppearance(const CurveAppearanceWidget::CurveAppearance &guiAppearance)
{
    BC::Data::CurveAppearance data;
    
    data.color = guiAppearance.color;
    data.curveStyle = fromQwtCurveStyle(guiAppearance.curveStyle);
    data.lineThickness = guiAppearance.lineThickness;
    data.lineStyle = fromQtPenStyle(guiAppearance.lineStyle);
    data.markerStyle = fromQwtSymbolStyle(guiAppearance.markerStyle);
    data.markerSize = guiAppearance.markerSize;
    data.visible = guiAppearance.visible;
    data.autoscale = guiAppearance.autoscale;
    data.yAxis = fromQwtAxisId(guiAppearance.yAxis);
    
    return data;
}

// =============================================================================
// Data to GUI enum conversions
// =============================================================================

QwtPlotCurve::CurveStyle CurveAppearanceConverter::toQwtCurveStyle(BC::Data::CurveStyle style)
{
    switch (style) {
        case BC::Data::CurveStyle::NoCurve: return QwtPlotCurve::NoCurve;
        case BC::Data::CurveStyle::Lines:   return QwtPlotCurve::Lines;
        case BC::Data::CurveStyle::Sticks:  return QwtPlotCurve::Sticks;
        case BC::Data::CurveStyle::Steps:   return QwtPlotCurve::Steps;
        case BC::Data::CurveStyle::Dots:    return QwtPlotCurve::Dots;
        default:                            return QwtPlotCurve::Lines;
    }
}

Qt::PenStyle CurveAppearanceConverter::toQtPenStyle(BC::Data::LineStyle style)
{
    switch (style) {
        case BC::Data::LineStyle::NoPen:           return Qt::NoPen;
        case BC::Data::LineStyle::SolidLine:       return Qt::SolidLine;
        case BC::Data::LineStyle::DashLine:        return Qt::DashLine;
        case BC::Data::LineStyle::DotLine:         return Qt::DotLine;
        case BC::Data::LineStyle::DashDotLine:     return Qt::DashDotLine;
        case BC::Data::LineStyle::DashDotDotLine:  return Qt::DashDotDotLine;
        default:                                   return Qt::SolidLine;
    }
}

QwtSymbol::Style CurveAppearanceConverter::toQwtSymbolStyle(BC::Data::MarkerStyle style)
{
    switch (style) {
        case BC::Data::MarkerStyle::NoSymbol:  return QwtSymbol::NoSymbol;
        case BC::Data::MarkerStyle::Ellipse:   return QwtSymbol::Ellipse;
        case BC::Data::MarkerStyle::Rect:      return QwtSymbol::Rect;
        case BC::Data::MarkerStyle::Diamond:   return QwtSymbol::Diamond;
        case BC::Data::MarkerStyle::Triangle:  return QwtSymbol::Triangle;
        case BC::Data::MarkerStyle::DTriangle: return QwtSymbol::DTriangle;
        case BC::Data::MarkerStyle::UTriangle: return QwtSymbol::UTriangle;
        case BC::Data::MarkerStyle::LTriangle: return QwtSymbol::LTriangle;
        case BC::Data::MarkerStyle::RTriangle: return QwtSymbol::RTriangle;
        case BC::Data::MarkerStyle::Cross:     return QwtSymbol::Cross;
        case BC::Data::MarkerStyle::XCross:    return QwtSymbol::XCross;
        case BC::Data::MarkerStyle::HLine:     return QwtSymbol::HLine;
        case BC::Data::MarkerStyle::VLine:     return QwtSymbol::VLine;
        case BC::Data::MarkerStyle::Star1:     return QwtSymbol::Star1;
        case BC::Data::MarkerStyle::Star2:     return QwtSymbol::Star2;
        case BC::Data::MarkerStyle::Hexagon:   return QwtSymbol::Hexagon;
        default:                               return QwtSymbol::NoSymbol;
    }
}

QwtAxisId CurveAppearanceConverter::toQwtAxisId(BC::Data::YAxis yAxis)
{
    switch (yAxis) {
        case BC::Data::YAxis::Left:  return QwtAxisId(QwtAxis::YLeft);
        case BC::Data::YAxis::Right: return QwtAxisId(QwtAxis::YRight);
        default:                    return QwtAxisId(QwtAxis::YLeft);
    }
}

// =============================================================================
// GUI to Data enum conversions
// =============================================================================

BC::Data::CurveStyle CurveAppearanceConverter::fromQwtCurveStyle(QwtPlotCurve::CurveStyle style)
{
    switch (style) {
        case QwtPlotCurve::NoCurve: return BC::Data::CurveStyle::NoCurve;
        case QwtPlotCurve::Lines:   return BC::Data::CurveStyle::Lines;
        case QwtPlotCurve::Sticks:  return BC::Data::CurveStyle::Sticks;
        case QwtPlotCurve::Steps:   return BC::Data::CurveStyle::Steps;
        case QwtPlotCurve::Dots:    return BC::Data::CurveStyle::Dots;
        default:                    return BC::Data::CurveStyle::Lines;
    }
}

BC::Data::LineStyle CurveAppearanceConverter::fromQtPenStyle(Qt::PenStyle style)
{
    switch (style) {
        case Qt::NoPen:           return BC::Data::LineStyle::NoPen;
        case Qt::SolidLine:       return BC::Data::LineStyle::SolidLine;
        case Qt::DashLine:        return BC::Data::LineStyle::DashLine;
        case Qt::DotLine:         return BC::Data::LineStyle::DotLine;
        case Qt::DashDotLine:     return BC::Data::LineStyle::DashDotLine;
        case Qt::DashDotDotLine:  return BC::Data::LineStyle::DashDotDotLine;
        default:                  return BC::Data::LineStyle::SolidLine;
    }
}

BC::Data::MarkerStyle CurveAppearanceConverter::fromQwtSymbolStyle(QwtSymbol::Style style)
{
    switch (style) {
        case QwtSymbol::NoSymbol:  return BC::Data::MarkerStyle::NoSymbol;
        case QwtSymbol::Ellipse:   return BC::Data::MarkerStyle::Ellipse;
        case QwtSymbol::Rect:      return BC::Data::MarkerStyle::Rect;
        case QwtSymbol::Diamond:   return BC::Data::MarkerStyle::Diamond;
        case QwtSymbol::Triangle:  return BC::Data::MarkerStyle::Triangle;
        case QwtSymbol::DTriangle: return BC::Data::MarkerStyle::DTriangle;
        case QwtSymbol::UTriangle: return BC::Data::MarkerStyle::UTriangle;
        case QwtSymbol::LTriangle: return BC::Data::MarkerStyle::LTriangle;
        case QwtSymbol::RTriangle: return BC::Data::MarkerStyle::RTriangle;
        case QwtSymbol::Cross:     return BC::Data::MarkerStyle::Cross;
        case QwtSymbol::XCross:    return BC::Data::MarkerStyle::XCross;
        case QwtSymbol::HLine:     return BC::Data::MarkerStyle::HLine;
        case QwtSymbol::VLine:     return BC::Data::MarkerStyle::VLine;
        case QwtSymbol::Star1:     return BC::Data::MarkerStyle::Star1;
        case QwtSymbol::Star2:     return BC::Data::MarkerStyle::Star2;
        case QwtSymbol::Hexagon:   return BC::Data::MarkerStyle::Hexagon;
        default:                   return BC::Data::MarkerStyle::NoSymbol;
    }
}

BC::Data::YAxis CurveAppearanceConverter::fromQwtAxisId(QwtAxisId axisId)
{
    if (axisId == QwtAxisId(QwtAxis::YRight)) {
        return BC::Data::YAxis::Right;
    } else {
        return BC::Data::YAxis::Left; // Default to left for any other axis
    }
}