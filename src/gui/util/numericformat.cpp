#include "numericformat.h"

#include <QRegularExpression>
#include <charconv>
#include <cmath>

#include <QtMath>

using namespace Qt::Literals::StringLiterals;

namespace BC::Gui {

QString formatScientificSuperscript(const QString &text)
{
    QString result = text;

    QRegularExpression ePattern("[eE]([+-]?\\d+)");
    QRegularExpressionMatch match = ePattern.match(result);

    if (match.hasMatch()) {
        QString exponent = match.captured(1);

        if (exponent.startsWith('+'))
            exponent = exponent.mid(1);

        const bool negative = exponent.startsWith('-');
        if (negative)
            exponent = exponent.mid(1);
        while (exponent.length() > 1 && exponent.startsWith('0'))
            exponent = exponent.mid(1);
        if (negative)
            exponent = '-' + exponent;

        if (exponent.startsWith('-'))
            result.replace(match.captured(0), u" \u00d7 10^(%1)"_s.arg(exponent));
        else
            result.replace(match.captured(0), u" \u00d7 10^%1"_s.arg(exponent));
    }

    return result;
}

QString formatNumberForDisplay(double value, int precision, NumericDisplayMode mode)
{
    if (qFuzzyIsNull(value))
        return u"0"_s;

    const double absValue = std::abs(value);
    const bool showFixed = (mode == NumericDisplayMode::Fixed) ||
                           (mode == NumericDisplayMode::Auto && absValue >= 1e-6 && absValue < 1e6);

    if (precision >= 0) {
        if (showFixed)
            return QString::number(value, 'f', precision);
        else
            return formatScientificSuperscript(QString::number(value, 'e', precision));
    }

    char buf[64];
    if (showFixed) {
        auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), value, std::chars_format::fixed);
        return QString::fromLatin1(buf, static_cast<int>(ptr - buf));
    }

    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), value, std::chars_format::scientific);
    return formatScientificSuperscript(QString::fromLatin1(buf, static_cast<int>(ptr - buf)));
}

} // namespace BC::Gui
