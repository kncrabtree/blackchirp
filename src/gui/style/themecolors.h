#ifndef THEMECOLORS_H
#define THEMECOLORS_H

#include <QColor>
#include <QWidget>
#include <QIcon>

/**
 * @brief Theme-aware color management system for BlackChirp
 * 
 * This class provides centralized color management that automatically adapts
 * to light/dark themes while maintaining accessibility and visual consistency.
 * All colors are derived from the system palette and adjusted for proper
 * contrast ratios.
 */
class ThemeColors
{
public:
    /**
     * @brief Color roles for different UI elements
     */
    enum ColorRole {
        // Status colors for feedback and states
        StatusSuccess,      // Success messages, valid states
        StatusWarning,      // Warning messages, caution states  
        StatusError,        // Error messages, invalid states
        StatusInfo,         // Information messages, neutral states
        StatusNeutral,      // Neutral/default status
        
        // Text colors for different emphasis levels
        SubtleText,         // Secondary text, less important content
        EmphasisText,       // Important text, highlighted content
        DisabledText,       // Disabled/inactive text
        
        // Icon colors for SVG theming
        IconPrimary,        // Primary icon color (main interface elements)
        IconSecondary,      // Secondary icon color (supporting elements)
        IconAccent          // Accent icon color (special highlighting)
    };

    /**
     * @brief Get a theme-aware color for the specified role
     * @param role The color role to get
     * @param widget Optional widget to use for palette context (default: nullptr)
     * @return QColor that adapts to the current theme
     */
    static QColor getThemeAwareColor(ColorRole role, const QWidget* widget = nullptr);

    /**
     * @brief Get a CSS color string for stylesheets
     * @param role The color role to get
     * @param widget Optional widget to use for palette context (default: nullptr)
     * @return QString in format suitable for CSS (e.g., "#ff0000" or "rgb(255,0,0)")
     */
    static QString getCSSColor(ColorRole role, const QWidget* widget = nullptr);

    /**
     * @brief Check if the current theme is dark
     * @param widget Optional widget to use for palette context (default: nullptr)
     * @return true if dark theme is detected, false for light theme
     */
    static bool isDarkTheme(const QWidget* widget = nullptr);

    /**
     * @brief Ensure adequate contrast for accessibility
     * @param color The color to adjust
     * @param background The background color to contrast against
     * @param minContrastRatio Minimum contrast ratio (default: 4.5 for WCAG AA)
     * @return QColor with adjusted contrast if necessary
     */
    static QColor ensureContrast(const QColor& color, const QColor& background, double minContrastRatio = 4.5);

    /**
     * @brief Calculate contrast ratio between two colors
     * @param color1 First color
     * @param color2 Second color
     * @return Contrast ratio (1.0 = no contrast, 21.0 = maximum contrast)
     */
    static double calculateContrastRatio(const QColor& color1, const QColor& color2);

    /**
     * @brief Create a theme-aware QIcon from an SVG resource
     * @param svgResourcePath Path to SVG resource (e.g., ":/icons/play.svg")
     * @param colorRole Color role to use for the icon (default: IconPrimary)
     * @param widget Optional widget for palette context (default: nullptr)
     * @return QIcon with theme-appropriate colors
     */
    static QIcon createThemedIcon(const QString& svgResourcePath, 
                                 ColorRole colorRole = IconPrimary, 
                                 const QWidget* widget = nullptr);

    /**
     * @brief Create a theme-aware QIcon with different colors for enabled/disabled states
     * @param svgResourcePath Path to SVG resource (e.g., ":/icons/play.svg")
     * @param enabledColorRole Color role for enabled state (default: IconPrimary)
     * @param disabledColorRole Color role for disabled state (default: DisabledText)
     * @param widget Optional widget for palette context (default: nullptr)
     * @return QIcon with proper enabled/disabled state colors
     */
    static QIcon createThemedIconWithStates(const QString& svgResourcePath,
                                           ColorRole enabledColorRole = IconPrimary,
                                           ColorRole disabledColorRole = DisabledText,
                                           const QWidget* widget = nullptr);

private:
    /**
     * @brief Get the base color for a role from the system palette
     * @param role The color role
     * @param widget Widget for palette context
     * @return Base QColor from system palette
     */
    static QColor getBaseColor(ColorRole role, const QWidget* widget);

    /**
     * @brief Adjust color brightness while preserving hue and saturation
     * @param color Original color
     * @param factor Brightness factor (-1.0 to 1.0, negative = darker, positive = lighter)
     * @return QColor with adjusted brightness
     */
    static QColor adjustBrightness(const QColor& color, double factor);

    /**
     * @brief Calculate relative luminance for contrast calculations
     * @param color The color to analyze
     * @return Relative luminance (0.0 = black, 1.0 = white)
     */
    static double relativeLuminance(const QColor& color);
};

#endif // THEMECOLORS_H