#ifndef CURVEAPPEARANCEPRESETMANAGER_H
#define CURVEAPPEARANCEPRESETMANAGER_H

#include <QObject>
#include <QDateTime>
#include <data/storage/settingsstorage.h>
#include "curveappearancewidget.h"

namespace BC::Key::CurveAppearancePresets {
inline constexpr QLatin1StringView key{"CurveAppearancePresets"};
inline constexpr QLatin1StringView presetList{"presetList"};
inline constexpr QLatin1StringView defaultsCreated{"defaultsCreated"};

// Default preset names
inline constexpr QLatin1StringView curvePrimary{"Curve - Primary"};
inline constexpr QLatin1StringView curveSecondary{"Curve - Secondary"};
inline constexpr QLatin1StringView curveTertiary{"Curve - Tertiary"};
inline constexpr QLatin1StringView stemPrimary{"Stem - Primary"};
inline constexpr QLatin1StringView stemSecondary{"Stem - Secondary"};
inline constexpr QLatin1StringView stemTertiary{"Stem - Tertiary"};
inline constexpr QLatin1StringView scatterCircles{"Scatter - Circles"};
inline constexpr QLatin1StringView scatterSquares{"Scatter - Squares"};
inline constexpr QLatin1StringView scatterDiamonds{"Scatter - Diamonds"};

// QVariantMap serialization keys
inline constexpr QLatin1StringView name{"name"};
inline constexpr QLatin1StringView created{"created"};
inline constexpr QLatin1StringView lastUsed{"lastUsed"};
inline constexpr QLatin1StringView isDefault{"isDefault"};
inline constexpr QLatin1StringView color{"color"};
inline constexpr QLatin1StringView curveStyle{"curveStyle"};
inline constexpr QLatin1StringView lineThickness{"lineThickness"};
inline constexpr QLatin1StringView lineStyle{"lineStyle"};
inline constexpr QLatin1StringView markerStyle{"markerStyle"};
inline constexpr QLatin1StringView markerSize{"markerSize"};
inline constexpr QLatin1StringView visible{"visible"};
inline constexpr QLatin1StringView autoscale{"autoscale"};
inline constexpr QLatin1StringView yAxis{"yAxis"};
}

class CurveAppearancePresetManager : public QObject, public SettingsStorage
{
    Q_OBJECT

public:
    // Singleton access
    static CurveAppearancePresetManager* instance();
    static void cleanup(); // Call during application shutdown
    
    struct CurveAppearancePreset {
        QString name;
        CurveAppearanceWidget::CurveAppearance appearance;
        QDateTime created;
        QDateTime lastUsed;
        bool isDefault;
        
        CurveAppearancePreset() : isDefault(false) {}
        
        // Convert to/from QVariantMap for storage
        QVariantMap toVariantMap() const;
        static CurveAppearancePreset fromVariantMap(const QVariantMap &map);
    };

    // Destructor
    ~CurveAppearancePresetManager();

    // Preset management
    QStringList getPresetNames() const;
    QList<CurveAppearancePreset> getAllPresets() const;
    CurveAppearancePreset getPreset(const QString &name) const;
    bool hasPreset(const QString &name) const;
    
    // Preset operations
    bool savePreset(const QString &name, const CurveAppearanceWidget::CurveAppearance &appearance, bool isDefault = false);
    bool deletePreset(const QString &name);
    bool renamePreset(const QString &oldName, const QString &newName);
    
    // Usage tracking
    void markPresetUsed(const QString &name);
    
    // Default preset creation
    void createDefaultPresets();
    bool areDefaultPresetsCreated() const;

signals:
    void presetSaved(const QString &name);
    void presetDeleted(const QString &name);
    void presetRenamed(const QString &oldName, const QString &newName);

private:
    // Private constructor for singleton
    explicit CurveAppearancePresetManager(QObject *parent = nullptr);
    
    // Static instance
    static CurveAppearancePresetManager* s_instance;
    void loadPresetsFromStorage();
    void savePresetsToStorage();
    CurveAppearanceWidget::CurveAppearance createCurvePreset(const QColor &color) const;
    CurveAppearanceWidget::CurveAppearance createStemPreset(const QColor &color) const;
    CurveAppearanceWidget::CurveAppearance createScatterPreset(const QColor &color, QwtSymbol::Style marker) const;
    
    QList<CurveAppearancePreset> d_presets;
    mutable QMap<QString, int> d_presetIndex; // Cache for quick lookup
    void rebuildIndex();
};

#endif // CURVEAPPEARANCEPRESETMANAGER_H