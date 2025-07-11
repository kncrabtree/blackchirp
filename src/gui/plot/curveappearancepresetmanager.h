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
}

class CurveAppearancePresetManager : public QObject, public SettingsStorage
{
    Q_OBJECT

public:
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

    explicit CurveAppearancePresetManager(QObject *parent = nullptr);
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