#include "overlaytypespecificwidget.h"

OverlayTypeSpecificWidget::OverlayTypeSpecificWidget(const Ft &currentFt, QWidget *parent)
    : QWidget(parent),
      d_context(Context::Creation),
      d_currentFt(currentFt)
{
    // Base class constructor - derived classes handle UI setup
}