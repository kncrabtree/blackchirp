#include "themecolors.h"

#include <QApplication>
#include <QPalette>
#include <QtMath>
#include <QFile>
#include <QPixmap>
#include <QByteArray>

QColor ThemeColors::getThemeAwareColor(ColorRole role, const QWidget* widget)
{
    QColor baseColor = getBaseColor(role, widget);
    bool darkTheme = isDarkTheme(widget);
    
    // Get background color for contrast calculations
    QPalette palette = widget ? widget->palette() : QApplication::palette();
    QColor background = palette.color(QPalette::Window);
    
    // Adjust color based on role and theme
    QColor adjustedColor = baseColor;
    
    switch (role) {
    case StatusSuccess:
        // Green tones - adjust for theme
        adjustedColor = darkTheme ? QColor(76, 175, 80) : QColor(46, 125, 50);
        break;
        
    case StatusWarning:
        // Orange/amber tones - adjust for theme
        adjustedColor = darkTheme ? QColor(255, 193, 7) : QColor(255, 152, 0);
        break;
        
    case StatusError:
        // Red tones - adjust for theme
        adjustedColor = darkTheme ? QColor(244, 67, 54) : QColor(211, 47, 47);
        break;
        
    case StatusInfo:
        // Blue tones - adjust for theme
        adjustedColor = darkTheme ? QColor(33, 150, 243) : QColor(25, 118, 210);
        break;
        
    case StatusNeutral:
        // Gray tones - use system text color with reduced opacity
        adjustedColor = palette.color(QPalette::Text);
        adjustedColor.setAlpha(128); // 50% opacity
        break;
        
    case SubtleText:
        // Secondary text - lighter than primary text
        adjustedColor = palette.color(QPalette::Text);
        adjustedColor = adjustBrightness(adjustedColor, darkTheme ? -0.3 : 0.3);
        break;
        
    case EmphasisText:
        // Emphasized text - use highlight color or enhanced text color
        if (palette.color(QPalette::Highlight).isValid()) {
            adjustedColor = palette.color(QPalette::Highlight);
        } else {
            adjustedColor = palette.color(QPalette::Text);
            adjustedColor = adjustBrightness(adjustedColor, darkTheme ? 0.2 : -0.2);
        }
        break;
        
    case DisabledText:
        // Disabled text - use system disabled text color
        adjustedColor = palette.color(QPalette::Disabled, QPalette::Text);
        break;
        
    case IconPrimary:
        // Primary icons - use text color for good contrast
        adjustedColor = palette.color(QPalette::Text);
        break;
        
    case IconSecondary:
        // Secondary icons - slightly muted
        adjustedColor = palette.color(QPalette::Text);
        adjustedColor = adjustBrightness(adjustedColor, darkTheme ? -0.2 : 0.2);
        break;
        
    case IconAccent:
        // Accent icons - use highlight color
        adjustedColor = palette.color(QPalette::Highlight);
        break;
    }
    
    // Ensure adequate contrast for accessibility
    adjustedColor = ensureContrast(adjustedColor, background);
    
    return adjustedColor;
}

QString ThemeColors::getCSSColor(ColorRole role, const QWidget* widget)
{
    QColor color = getThemeAwareColor(role, widget);
    return color.name(); // Returns hex format like "#ff0000"
}

bool ThemeColors::isDarkTheme(const QWidget* widget)
{
    QPalette palette = widget ? widget->palette() : QApplication::palette();
    
    // Calculate luminance of window background
    QColor background = palette.color(QPalette::Window);
    double luminance = relativeLuminance(background);
    
    // Threshold for dark theme detection (0.5 = 50% gray)
    return luminance < 0.5;
}

QColor ThemeColors::ensureContrast(const QColor& color, const QColor& background, double minContrastRatio)
{
    double currentRatio = calculateContrastRatio(color, background);
    
    if (currentRatio >= minContrastRatio) {
        return color; // Already sufficient contrast
    }
    
    // Adjust brightness to meet contrast requirements
    QColor adjusted = color;
    double backgroundLuminance = relativeLuminance(background);
    
    // Determine if we should make the color lighter or darker
    // Dark background (low luminance) needs lighter colors for contrast
    bool shouldLighten = backgroundLuminance < 0.5;
    
    // Binary search for optimal brightness adjustment
    double minFactor = shouldLighten ? 0.0 : -1.0;
    double maxFactor = shouldLighten ? 1.0 : 0.0;
    double factor = shouldLighten ? 0.5 : -0.5;
    
    for (int i = 0; i < 10; ++i) { // Max 10 iterations
        adjusted = adjustBrightness(color, factor);
        double ratio = calculateContrastRatio(adjusted, background);
        
        if (qAbs(ratio - minContrastRatio) < 0.1) {
            break; // Close enough
        }
        
        if (ratio < minContrastRatio) {
            if (shouldLighten) {
                minFactor = factor;
                factor = (factor + maxFactor) / 2.0;
            } else {
                maxFactor = factor;
                factor = (minFactor + factor) / 2.0;
            }
        } else {
            if (shouldLighten) {
                maxFactor = factor;
                factor = (minFactor + factor) / 2.0;
            } else {
                minFactor = factor;
                factor = (factor + maxFactor) / 2.0;
            }
        }
    }
    
    return adjusted;
}

double ThemeColors::calculateContrastRatio(const QColor& color1, const QColor& color2)
{
    double lum1 = relativeLuminance(color1);
    double lum2 = relativeLuminance(color2);
    
    // Ensure lum1 is the lighter color
    if (lum1 < lum2) {
        qSwap(lum1, lum2);
    }
    
    return (lum1 + 0.05) / (lum2 + 0.05);
}

QColor ThemeColors::getBaseColor(ColorRole role, const QWidget* widget)
{
    QPalette palette = widget ? widget->palette() : QApplication::palette();
    
    // Return appropriate palette color based on role
    switch (role) {
    case StatusSuccess:
    case StatusWarning:
    case StatusError:
    case StatusInfo:
        // Status colors will be handled in getThemeAwareColor
        return QColor();
        
    case StatusNeutral:
    case SubtleText:
    case EmphasisText:
    case IconPrimary:
    case IconSecondary:
        return palette.color(QPalette::Text);
        
    case DisabledText:
        return palette.color(QPalette::Disabled, QPalette::Text);
        
    case IconAccent:
        return palette.color(QPalette::Highlight);
    }
    
    return palette.color(QPalette::Text); // Default fallback
}

QColor ThemeColors::adjustBrightness(const QColor& color, double factor)
{
    // Clamp factor to valid range
    factor = qBound(-1.0, factor, 1.0);
    
    // Convert to HSV for brightness adjustment
    int h, s, v, a;
    color.getHsv(&h, &s, &v, &a);
    
    if (factor > 0) {
        // Lighten: move towards white (value = 255)
        v = qRound(v + (255 - v) * factor);
    } else {
        // Darken: move towards black (value = 0)
        v = qRound(v * (1.0 + factor));
    }
    
    v = qBound(0, v, 255);
    
    return QColor::fromHsv(h, s, v, a);
}

double ThemeColors::relativeLuminance(const QColor& color)
{
    // Convert RGB to linear RGB
    auto linearRGB = [](double c) {
        c = c / 255.0;
        return (c <= 0.03928) ? c / 12.92 : qPow((c + 0.055) / 1.055, 2.4);
    };
    
    double r = linearRGB(color.red());
    double g = linearRGB(color.green());
    double b = linearRGB(color.blue());
    
    // Calculate relative luminance using ITU-R BT.709 coefficients
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

QIcon ThemeColors::createThemedIcon(const QString& svgResourcePath, ColorRole colorRole, const QWidget* widget)
{
    return createThemedIconWithStates(svgResourcePath, colorRole, DisabledText, widget);
}

QIcon ThemeColors::createThemedIconWithStates(const QString& svgResourcePath, 
                                              ColorRole enabledColorRole, 
                                              ColorRole disabledColorRole, 
                                              const QWidget* widget)
{
    // Helper function to create a colored pixmap from SVG content
    auto createColoredPixmap = [](const QString& svgContent, const QString& color) -> QPixmap {
        QString modifiedSvg = svgContent;
        
        // Replace various ways SVG might specify black/currentColor with our theme color
        modifiedSvg.replace("currentColor", color);
        modifiedSvg.replace("fill=\"#000000\"", QString("fill=\"%1\"").arg(color));
        modifiedSvg.replace("fill=\"#000\"", QString("fill=\"%1\"").arg(color)); 
        modifiedSvg.replace("fill=\"black\"", QString("fill=\"%1\"").arg(color));
        modifiedSvg.replace("stroke=\"#000000\"", QString("stroke=\"%1\"").arg(color));
        modifiedSvg.replace("stroke=\"#000\"", QString("stroke=\"%1\"").arg(color));
        modifiedSvg.replace("stroke=\"black\"", QString("stroke=\"%1\"").arg(color));
        
        // Also handle cases where no fill is specified (defaults to black)
        if (!modifiedSvg.contains("fill=")) {
            modifiedSvg.replace("<path", QString("<path fill=\"%1\"").arg(color));
            modifiedSvg.replace("<circle", QString("<circle fill=\"%1\"").arg(color));
            modifiedSvg.replace("<rect", QString("<rect fill=\"%1\"").arg(color));
            modifiedSvg.replace("<ellipse", QString("<ellipse fill=\"%1\"").arg(color));
            modifiedSvg.replace("<polygon", QString("<polygon fill=\"%1\"").arg(color));
        }
        
        QByteArray svgData = modifiedSvg.toUtf8();
        QPixmap pixmap;
        pixmap.loadFromData(svgData, "SVG");
        return pixmap;
    };
    
    // Read the SVG file
    QFile file(svgResourcePath);
    if (!file.open(QIODevice::ReadOnly)) {
        // Return empty icon if file can't be read
        return QIcon();
    }
    
    QString svgContent = QString::fromUtf8(file.readAll());
    file.close();
    
    // Get theme colors
    QString enabledColor = getCSSColor(enabledColorRole, widget);
    QString disabledColor = getCSSColor(disabledColorRole, widget);
    
    // Create icon with different states
    QIcon icon;
    
    // Normal (enabled) state
    QPixmap enabledPixmap = createColoredPixmap(svgContent, enabledColor);
    icon.addPixmap(enabledPixmap, QIcon::Normal, QIcon::Off);
    icon.addPixmap(enabledPixmap, QIcon::Normal, QIcon::On);
    icon.addPixmap(enabledPixmap, QIcon::Active, QIcon::Off);
    icon.addPixmap(enabledPixmap, QIcon::Active, QIcon::On);
    icon.addPixmap(enabledPixmap, QIcon::Selected, QIcon::Off);
    icon.addPixmap(enabledPixmap, QIcon::Selected, QIcon::On);
    
    // Disabled state
    QPixmap disabledPixmap = createColoredPixmap(svgContent, disabledColor);
    icon.addPixmap(disabledPixmap, QIcon::Disabled, QIcon::Off);
    icon.addPixmap(disabledPixmap, QIcon::Disabled, QIcon::On);
    
    return icon;
}