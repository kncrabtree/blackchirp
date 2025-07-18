#ifndef CURVEAPPEARANCEPRESETMANAGER_H
#define CURVEAPPEARANCEPRESETMANAGER_H

#include <QObject>
#include <QDateTime>
#include <data/storage/settingsstorage.h>
#include "curveappearancewidget.h"

namespace BC::Key::CurveAppearancePresets {
static const QString key{"CurveAppearancePresets"};
static const QString presetList{"presetList"};
static const QString defaultsCreated{"defaultsCreated"};

// Default preset names
static const QString curvePrimary{"Curve - Primary"};
static const QString curveSecondary{"Curve - Secondary"};
static const QString curveTertiary{"Curve - Tertiary"};
static const QString stemPrimary{"Stem - Primary"};
static const QString stemSecondary{"Stem - Secondary"};
static const QString stemTertiary{"Stem - Tertiary"};
static const QString scatterCircles{"Scatter - Circles"};
static const QString scatterSquares{"Scatter - Squares"};
static const QString scatterDiamonds{"Scatter - Diamonds"};

// QVariantMap serialization keys
static const QString name{"name"};
static const QString created{"created"};
static const QString lastUsed{"lastUsed"};
static const QString isDefault{"isDefault"};
static const QString color{"color"};
static const QString curveStyle{"curveStyle"};
static const QString lineThickness{"lineThickness"};
static const QString lineStyle{"lineStyle"};
static const QString markerStyle{"markerStyle"};
static const QString markerSize{"markerSize"};
static const QString visible{"visible"};
static const QString autoscale{"autoscale"};
static const QString yAxis{"yAxis"};
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