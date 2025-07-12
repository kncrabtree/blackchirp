#include "overlaytypespecificwidget.h"

OverlayTypeSpecificWidget::OverlayTypeSpecificWidget(QWidget *parent)
    : QWidget(parent),
      d_context(Context::Creation)
{
    // Base class constructor - derived classes handle UI setup
}