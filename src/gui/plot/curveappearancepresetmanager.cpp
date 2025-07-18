#include "curveappearancepresetmanager.h"
#include <QApplication>
#include <QPalette>
#include <QDebug>

// Static instance definition
CurveAppearancePresetManager* CurveAppearancePresetManager::s_instance = nullptr;

CurveAppearancePresetManager::CurveAppearancePresetManager(QObject *parent)
    : QObject(parent), SettingsStorage(BC::Key::CurveAppearancePresets::key)
{
    loadPresetsFromStorage();
    
    // Create default presets if they don't exist
    if (!areDefaultPresetsCreated()) {
        createDefaultPresets();
        set(BC::Key::CurveAppearancePresets::defaultsCreated, true, true);
    }
}

CurveAppearancePresetManager::~CurveAppearancePresetManager()
{
    savePresetsToStorage();
}

// === SINGLETON IMPLEMENTATION ===

CurveAppearancePresetManager* CurveAppearancePresetManager::instance()
{
    if (!s_instance) {
        s_instance = new CurveAppearancePresetManager();
    }
    return s_instance;
}

void CurveAppearancePresetManager::cleanup()
{
    if (s_instance) {
        delete s_instance;
        s_instance = nullptr;
    }
}

QVariantMap CurveAppearancePresetManager::CurveAppearancePreset::toVariantMap() const
{
    QVariantMap map;
    map[BC::Key::CurveAppearancePresets::name] = name;
    map[BC::Key::CurveAppearancePresets::created] = created;
    map[BC::Key::CurveAppearancePresets::lastUsed] = lastUsed;
    map[BC::Key::CurveAppearancePresets::isDefault] = isDefault;
    
    // Store appearance properties
    map[BC::Key::CurveAppearancePresets::color] = appearance.color;
    map[BC::Key::CurveAppearancePresets::curveStyle] = static_cast<int>(appearance.curveStyle);
    map[BC::Key::CurveAppearancePresets::lineThickness] = appearance.lineThickness;
    map[BC::Key::CurveAppearancePresets::lineStyle] = static_cast<int>(appearance.lineStyle);
    map[BC::Key::CurveAppearancePresets::markerStyle] = static_cast<int>(appearance.markerStyle);
    map[BC::Key::CurveAppearancePresets::markerSize] = appearance.markerSize;
    map[BC::Key::CurveAppearancePresets::visible] = appearance.visible;
    map[BC::Key::CurveAppearancePresets::autoscale] = appearance.autoscale;
    map[BC::Key::CurveAppearancePresets::yAxis] = static_cast<int>(appearance.yAxis);
    
    return map;
}

CurveAppearancePresetManager::CurveAppearancePreset CurveAppearancePresetManager::CurveAppearancePreset::fromVariantMap(const QVariantMap &map)
{
    CurveAppearancePreset preset;
    preset.name = map.value(BC::Key::CurveAppearancePresets::name).toString();
    preset.created = map.value(BC::Key::CurveAppearancePresets::created).toDateTime();
    preset.lastUsed = map.value(BC::Key::CurveAppearancePresets::lastUsed).toDateTime();
    preset.isDefault = map.value(BC::Key::CurveAppearancePresets::isDefault, false).toBool();
    
    // Load appearance properties
    preset.appearance.color = map.value(BC::Key::CurveAppearancePresets::color, QColor(Qt::black)).value<QColor>();
    preset.appearance.curveStyle = static_cast<QwtPlotCurve::CurveStyle>(map.value(BC::Key::CurveAppearancePresets::curveStyle, QwtPlotCurve::Lines).toInt());
    preset.appearance.lineThickness = map.value(BC::Key::CurveAppearancePresets::lineThickness, 1.0).toDouble();
    preset.appearance.lineStyle = static_cast<Qt::PenStyle>(map.value(BC::Key::CurveAppearancePresets::lineStyle, static_cast<int>(Qt::SolidLine)).toInt());
    preset.appearance.markerStyle = static_cast<QwtSymbol::Style>(map.value(BC::Key::CurveAppearancePresets::markerStyle, QwtSymbol::NoSymbol).toInt());
    preset.appearance.markerSize = map.value(BC::Key::CurveAppearancePresets::markerSize, 7).toInt();
    preset.appearance.visible = map.value(BC::Key::CurveAppearancePresets::visible, true).toBool();
    preset.appearance.autoscale = map.value(BC::Key::CurveAppearancePresets::autoscale, true).toBool();
    preset.appearance.yAxis = static_cast<QwtAxisId>(map.value(BC::Key::CurveAppearancePresets::yAxis, QwtAxis::YLeft).toInt());
    
    return preset;
}

QStringList CurveAppearancePresetManager::getPresetNames() const
{
    QStringList names;
    for (const auto &preset : d_presets) {
        names << preset.name;
    }
    return names;
}

QList<CurveAppearancePresetManager::CurveAppearancePreset> CurveAppearancePresetManager::getAllPresets() const
{
    return d_presets;
}

CurveAppearancePresetManager::CurveAppearancePreset CurveAppearancePresetManager::getPreset(const QString &name) const
{
    if (d_presetIndex.contains(name)) {
        int index = d_presetIndex[name];
        if (index >= 0 && index < d_presets.size()) {
            return d_presets[index];
        }
    }
    return CurveAppearancePreset(); // Return empty preset if not found
}

bool CurveAppearancePresetManager::hasPreset(const QString &name) const
{
    return d_presetIndex.contains(name);
}

bool CurveAppearancePresetManager::savePreset(const QString &name, const CurveAppearanceWidget::CurveAppearance &appearance, bool isDefault)
{
    if (name.isEmpty()) {
        return false;
    }
    
    CurveAppearancePreset preset;
    preset.name = name;
    preset.appearance = appearance;
    preset.created = QDateTime::currentDateTime();
    preset.lastUsed = preset.created;
    preset.isDefault = isDefault;
    
    // Check if preset already exists
    if (d_presetIndex.contains(name)) {
        // Update existing preset (but preserve creation date if it's a default)
        int index = d_presetIndex[name];
        if (index >= 0 && index < d_presets.size()) {
            preset.created = d_presets[index].created;
            preset.isDefault = d_presets[index].isDefault;
            d_presets[index] = preset;
        }
    } else {
        // Add new preset
        d_presets.append(preset);
        rebuildIndex();
    }
    
    savePresetsToStorage();
    emit presetSaved(name);
    
    return true;
}

bool CurveAppearancePresetManager::deletePreset(const QString &name)
{
    if (!d_presetIndex.contains(name)) {
        return false;
    }
    
    int index = d_presetIndex[name];
    if (index >= 0 && index < d_presets.size()) {
        // Don't allow deletion of default presets
        if (d_presets[index].isDefault) {
            qWarning() << "Cannot delete default preset:" << name;
            return false;
        }
        
        d_presets.removeAt(index);
        rebuildIndex();
        savePresetsToStorage();
        emit presetDeleted(name);
        
        return true;
    }
    
    return false;
}

bool CurveAppearancePresetManager::renamePreset(const QString &oldName, const QString &newName)
{
    if (oldName.isEmpty() || newName.isEmpty() || oldName == newName) {
        return false;
    }
    
    if (!d_presetIndex.contains(oldName) || d_presetIndex.contains(newName)) {
        return false; // Old name doesn't exist or new name already exists
    }
    
    int index = d_presetIndex[oldName];
    if (index >= 0 && index < d_presets.size()) {
        // Don't allow renaming of default presets
        if (d_presets[index].isDefault) {
            qWarning() << "Cannot rename default preset:" << oldName;
            return false;
        }
        
        d_presets[index].name = newName;
        rebuildIndex();
        savePresetsToStorage();
        emit presetRenamed(oldName, newName);
        
        return true;
    }
    
    return false;
}

void CurveAppearancePresetManager::markPresetUsed(const QString &name)
{
    if (d_presetIndex.contains(name)) {
        int index = d_presetIndex[name];
        if (index >= 0 && index < d_presets.size()) {
            d_presets[index].lastUsed = QDateTime::currentDateTime();
            savePresetsToStorage();
        }
    }
}

void CurveAppearancePresetManager::createDefaultPresets()
{    
    // Get colors from QPalette for consistency with application theming
    QPalette palette = QApplication::palette();
    QColor primaryColor = palette.color(QPalette::Highlight);
    QColor secondaryColor = palette.color(QPalette::Link);
    QColor tertiaryColor = palette.color(QPalette::LinkVisited);
    
    // Ensure colors are visible - fallback to standard colors if palette colors are too similar
    if (primaryColor.lightness() < 50) {
        primaryColor = QColor(Qt::blue);
        secondaryColor = QColor(Qt::red);
        tertiaryColor = QColor(Qt::darkGreen);
    }
    
    QDateTime now = QDateTime::currentDateTime();
    
    // === CURVE PRESETS (3) ===
    CurveAppearancePreset curvePreset1;
    curvePreset1.name = BC::Key::CurveAppearancePresets::curvePrimary;
    curvePreset1.appearance = createCurvePreset(primaryColor);
    curvePreset1.created = now;
    curvePreset1.lastUsed = now;
    curvePreset1.isDefault = true;
    d_presets.append(curvePreset1);
    
    CurveAppearancePreset curvePreset2;
    curvePreset2.name = BC::Key::CurveAppearancePresets::curveSecondary;
    curvePreset2.appearance = createCurvePreset(secondaryColor);
    curvePreset2.created = now;
    curvePreset2.lastUsed = now;
    curvePreset2.isDefault = true;
    d_presets.append(curvePreset2);
    
    CurveAppearancePreset curvePreset3;
    curvePreset3.name = BC::Key::CurveAppearancePresets::curveTertiary;
    curvePreset3.appearance = createCurvePreset(tertiaryColor);
    curvePreset3.created = now;
    curvePreset3.lastUsed = now;
    curvePreset3.isDefault = true;
    d_presets.append(curvePreset3);
    
    // === STEM PRESETS (3) ===
    CurveAppearancePreset stemPreset1;
    stemPreset1.name = BC::Key::CurveAppearancePresets::stemPrimary;
    stemPreset1.appearance = createStemPreset(primaryColor);
    stemPreset1.created = now;
    stemPreset1.lastUsed = now;
    stemPreset1.isDefault = true;
    d_presets.append(stemPreset1);
    
    CurveAppearancePreset stemPreset2;
    stemPreset2.name = BC::Key::CurveAppearancePresets::stemSecondary;
    stemPreset2.appearance = createStemPreset(secondaryColor);
    stemPreset2.created = now;
    stemPreset2.lastUsed = now;
    stemPreset2.isDefault = true;
    d_presets.append(stemPreset2);
    
    CurveAppearancePreset stemPreset3;
    stemPreset3.name = BC::Key::CurveAppearancePresets::stemTertiary;
    stemPreset3.appearance = createStemPreset(tertiaryColor);
    stemPreset3.created = now;
    stemPreset3.lastUsed = now;
    stemPreset3.isDefault = true;
    d_presets.append(stemPreset3);
    
    // === SCATTER PRESETS (3) ===
    CurveAppearancePreset scatterPreset1;
    scatterPreset1.name = BC::Key::CurveAppearancePresets::scatterCircles;
    scatterPreset1.appearance = createScatterPreset(primaryColor, QwtSymbol::Ellipse);
    scatterPreset1.created = now;
    scatterPreset1.lastUsed = now;
    scatterPreset1.isDefault = true;
    d_presets.append(scatterPreset1);
    
    CurveAppearancePreset scatterPreset2;
    scatterPreset2.name = BC::Key::CurveAppearancePresets::scatterSquares;
    scatterPreset2.appearance = createScatterPreset(secondaryColor, QwtSymbol::Rect);
    scatterPreset2.created = now;
    scatterPreset2.lastUsed = now;
    scatterPreset2.isDefault = true;
    d_presets.append(scatterPreset2);
    
    CurveAppearancePreset scatterPreset3;
    scatterPreset3.name = BC::Key::CurveAppearancePresets::scatterDiamonds;
    scatterPreset3.appearance = createScatterPreset(tertiaryColor, QwtSymbol::Diamond);
    scatterPreset3.created = now;
    scatterPreset3.lastUsed = now;
    scatterPreset3.isDefault = true;
    d_presets.append(scatterPreset3);
    
    rebuildIndex();
    savePresetsToStorage();
    
}

bool CurveAppearancePresetManager::areDefaultPresetsCreated() const
{
    return get(BC::Key::CurveAppearancePresets::defaultsCreated, false);
}

void CurveAppearancePresetManager::loadPresetsFromStorage()
{
    d_presets.clear();
    
    QStringList presetNames = get(BC::Key::CurveAppearancePresets::presetList, QStringList());
    
    for (const QString &name : presetNames) {
        QVariantMap presetData = get(name, QVariantMap());
        if (!presetData.isEmpty()) {
            CurveAppearancePreset preset = CurveAppearancePreset::fromVariantMap(presetData);
            if (!preset.name.isEmpty()) {
                d_presets.append(preset);
            }
        }
    }
    
    rebuildIndex();
}

void CurveAppearancePresetManager::savePresetsToStorage()
{
    QStringList presetNames;
    
    for (const auto &preset : d_presets) {
        presetNames << preset.name;
        set(preset.name, preset.toVariantMap(), false);
    }
    
    set(BC::Key::CurveAppearancePresets::presetList, presetNames, true);
}

CurveAppearanceWidget::CurveAppearance CurveAppearancePresetManager::createCurvePreset(const QColor &color) const
{
    CurveAppearanceWidget::CurveAppearance appearance;
    appearance.color = color;
    appearance.curveStyle = QwtPlotCurve::Lines;
    appearance.lineThickness = 1.0;
    appearance.lineStyle = Qt::SolidLine;
    appearance.markerStyle = QwtSymbol::NoSymbol;
    appearance.markerSize = 7;
    appearance.visible = true;
    appearance.autoscale = true;
    appearance.yAxis = QwtAxis::YLeft;
    return appearance;
}

CurveAppearanceWidget::CurveAppearance CurveAppearancePresetManager::createStemPreset(const QColor &color) const
{
    CurveAppearanceWidget::CurveAppearance appearance;
    appearance.color = color;
    appearance.curveStyle = QwtPlotCurve::Sticks;
    appearance.lineThickness = 2.0;
    appearance.lineStyle = Qt::SolidLine;
    appearance.markerStyle = QwtSymbol::NoSymbol;
    appearance.markerSize = 7;
    appearance.visible = true;
    appearance.autoscale = true;
    appearance.yAxis = QwtAxis::YLeft;
    return appearance;
}

CurveAppearanceWidget::CurveAppearance CurveAppearancePresetManager::createScatterPreset(const QColor &color, QwtSymbol::Style marker) const
{
    CurveAppearanceWidget::CurveAppearance appearance;
    appearance.color = color;
    appearance.curveStyle = QwtPlotCurve::NoCurve;
    appearance.lineThickness = 1.0;
    appearance.lineStyle = Qt::NoPen;
    appearance.markerStyle = marker;
    appearance.markerSize = 8;
    appearance.visible = true;
    appearance.autoscale = true;
    appearance.yAxis = QwtAxis::YLeft;
    return appearance;
}

void CurveAppearancePresetManager::rebuildIndex()
{
    d_presetIndex.clear();
    for (int i = 0; i < d_presets.size(); ++i) {
        d_presetIndex[d_presets[i].name] = i;
    }
}

