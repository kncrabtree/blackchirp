#ifndef NUMERICFORMAT_H
#define NUMERICFORMAT_H

#include <QString>

namespace BC::Gui {

enum class NumericDisplayMode { Auto, Fixed, Scientific };

QString formatScientificSuperscript(const QString &text);
QString formatNumberForDisplay(double value, int precision = -1,
                               NumericDisplayMode mode = NumericDisplayMode::Auto);

} // namespace BC::Gui

#endif // NUMERICFORMAT_H
