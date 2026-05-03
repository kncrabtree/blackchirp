#ifndef THEMECOLORS_H
#define THEMECOLORS_H

#include <QColor>
#include <QWidget>
#include <QIcon>

/// \brief All-static utility class for theme-aware colors and icons.
///
/// Every color is derived from the system palette and adjusted for
/// contrast so that status indicators and icons remain readable under
/// both light and dark system themes.
///
/// \sa ColorRole
class ThemeColors
{
public:
    /// \brief Semantic color roles used throughout the UI.
    enum ColorRole {
        // Status colors for feedback and states
        StatusSuccess,  ///< Success messages and valid states.
        StatusWarning,  ///< Warning messages and caution states.
        StatusError,    ///< Error messages and invalid states.
        StatusInfo,     ///< Informational messages and neutral states.
        StatusNeutral,  ///< Default or unclassified status.

        // Text colors for different emphasis levels
        SubtleText,     ///< Secondary or less-important text.
        EmphasisText,   ///< Important or highlighted text.
        DisabledText,   ///< Inactive or disabled text.

        // Icon colors for SVG theming
        IconPrimary,    ///< Main interface element icons.
        IconSecondary,  ///< Supporting element icons.
        IconAccent      ///< Special-highlight icons.
    };

    /// \brief Returns a palette-derived color for \a role.
    /// \param role   Color role to resolve.
    /// \param widget Widget whose palette is used as context; pass \c nullptr
    ///               to use the application palette.
    /// \return QColor appropriate for the active theme.
    static QColor getThemeAwareColor(ColorRole role, const QWidget* widget = nullptr);

    /// \brief Returns a CSS hex string for \a role (e.g., \c "#ff0000").
    /// \param role   Color role to resolve.
    /// \param widget Widget whose palette is used as context; pass \c nullptr
    ///               to use the application palette.
    /// \return QString suitable for use in a Qt stylesheet.
    static QString getCSSColor(ColorRole role, const QWidget* widget = nullptr);

    /// \brief Returns \c true when the active palette is a dark theme.
    /// \param widget Widget whose palette is inspected; pass \c nullptr for
    ///               the application palette.
    static bool isDarkTheme(const QWidget* widget = nullptr);

    /// \brief Adjusts \a color until its contrast ratio against \a background
    ///        meets \a minContrastRatio.
    ///
    /// The default target of 4.5 corresponds to WCAG AA compliance for normal
    /// text. Returns \a color unchanged if the ratio is already satisfied.
    ///
    /// \param color            Color to adjust.
    /// \param background       Background color to contrast against.
    /// \param minContrastRatio Minimum acceptable contrast ratio (default: 4.5).
    /// \return Adjusted QColor.
    static QColor ensureContrast(const QColor& color, const QColor& background, double minContrastRatio = 4.5);

    /// \brief Calculates the WCAG contrast ratio between two colors.
    ///
    /// The ratio ranges from 1.0 (identical colors) to 21.0 (black on white).
    ///
    /// \param color1 First color.
    /// \param color2 Second color.
    /// \return Contrast ratio.
    static double calculateContrastRatio(const QColor& color1, const QColor& color2);

    /// \brief Creates a QIcon from an SVG resource with the color for \a colorRole.
    /// \param svgResourcePath Qt resource path to the SVG (e.g., \c ":/icons/play.svg").
    /// \param colorRole       Color role applied to the icon (default: IconPrimary).
    /// \param widget          Widget for palette context (default: \c nullptr).
    /// \return QIcon with theme-appropriate coloring.
    static QIcon createThemedIcon(const QString& svgResourcePath,
                                 ColorRole colorRole = IconPrimary,
                                 const QWidget* widget = nullptr);

    /// \brief Creates a QIcon with separate colors for enabled and disabled states.
    /// \param svgResourcePath    Qt resource path to the SVG.
    /// \param enabledColorRole   Color role for the enabled state (default: IconPrimary).
    /// \param disabledColorRole  Color role for the disabled state (default: DisabledText).
    /// \param widget             Widget for palette context (default: \c nullptr).
    /// \return QIcon with per-state theme-appropriate coloring.
    static QIcon createThemedIconWithStates(const QString& svgResourcePath,
                                           ColorRole enabledColorRole = IconPrimary,
                                           ColorRole disabledColorRole = DisabledText,
                                           const QWidget* widget = nullptr);

private:
    /// \brief Returns the base palette-derived color for \a role.
    static QColor getBaseColor(ColorRole role, const QWidget* widget);

    /// \brief Adjusts the brightness of \a color by \a factor.
    ///
    /// \a factor ranges from -1.0 (fully dark) to 1.0 (fully bright).
    static QColor adjustBrightness(const QColor& color, double factor);

    /// \brief Computes the WCAG relative luminance of \a color.
    ///
    /// Returns a value in [0.0, 1.0] where 0.0 is black and 1.0 is white.
    static double relativeLuminance(const QColor& color);
};

#endif // THEMECOLORS_H
