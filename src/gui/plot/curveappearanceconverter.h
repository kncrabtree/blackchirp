#ifndef CURVEAPPEARANCECONVERTER_H
#define CURVEAPPEARANCECONVERTER_H

#include <data/presentation/curveappearance.h>
#include <gui/plot/curveappearancewidget.h>
#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_symbol.h>
#include <qwt6/qwt_axis_id.h>
#include <QPen>

/**
 * @brief Conversion utilities between data and GUI curve appearance structures
 * 
 * This class provides static functions to convert between the pure data structure
 * BC::Data::CurveAppearance and the GUI-dependent CurveAppearanceWidget::CurveAppearance.
 * 
 * The data layer uses custom enums to avoid GUI dependencies, while the GUI layer
 * uses Qt/QWT enums directly. These functions handle the mapping between them.
 */
class CurveAppearanceConverter
{
public:
    // Convert from data layer to GUI layer
    static CurveAppearanceWidget::CurveAppearance toGuiAppearance(const BC::Data::CurveAppearance &dataAppearance);
    
    // Convert from GUI layer to data layer  
    static BC::Data::CurveAppearance toDataAppearance(const CurveAppearanceWidget::CurveAppearance &guiAppearance);
    
    // Individual enum conversions - Data to GUI
    static QwtPlotCurve::CurveStyle toQwtCurveStyle(BC::Data::CurveStyle style);
    static Qt::PenStyle toQtPenStyle(BC::Data::LineStyle style);
    static QwtSymbol::Style toQwtSymbolStyle(BC::Data::MarkerStyle style);
    static QwtAxisId toQwtAxisId(BC::Data::YAxis yAxis);
    
    // Individual enum conversions - GUI to Data
    static BC::Data::CurveStyle fromQwtCurveStyle(QwtPlotCurve::CurveStyle style);
    static BC::Data::LineStyle fromQtPenStyle(Qt::PenStyle style);
    static BC::Data::MarkerStyle fromQwtSymbolStyle(QwtSymbol::Style style);
    static BC::Data::YAxis fromQwtAxisId(QwtAxisId axisId);

private:
    // Private constructor - this is a utility class with only static methods
    CurveAppearanceConverter() = delete;
};

#endif // CURVEAPPEARANCECONVERTER_H