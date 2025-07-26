#ifndef OVERLAYBASE_H
#define OVERLAYBASE_H

#include <map>
#include <QString>
#include <QPointF>
#include <QVariant>
#include <data/presentation/curveappearance.h>

class OverlayMetadataStorage;

namespace BC::Key::Overlay {
static const QString oLabel{"label"};
static const QString oSourceFile{"sourceFile"};
static const QString oDestFile{"destFile"};
static const QString oPlotId{"plotId"};
static const QString oYScale{"yScale"};
static const QString oYOffset{"yOffset"};
static const QString oXOffset{"xOffset"};
static const QString oMinFreqEnabled{"minFreqEnabled"};
static const QString oMinFreqValue{"minFreqValue"};
static const QString oMaxFreqEnabled{"maxFreqEnabled"};
static const QString oMaxFreqValue{"maxFreqValue"};
static const QString oEnabled{"enabled"};
static const QString oComment{"comment"};
static const QString overlaySettingsFile{"%1.settings.csv"};
static const QString overlayDataFile{"%1.data.csv"};
}

class OverlayBase
{
    Q_GADGET
    friend class OverlayStorage;
    friend class OverlayMetadataStorage;
    friend class UnifiedOverlayWidget;
    
public:
    enum OverlayType {
        BCExperiment,
        Catalog,
        GenericXY
    };
    Q_ENUM(OverlayType)

    OverlayBase(OverlayType type);
    
    
    QVector<QPointF> xyData() const;
    
    QString getLabel() const;
    QString getSourceFile() const;
    QString getDestFile() const;
    QString getPlotId() const;
    QString getComment() const;
    double getYScale() const;
    double getYOffset() const;
    double getXOffset() const;
    bool getMinFreqEnabled() const;
    double getMinFreqValue() const;
    bool getMaxFreqEnabled() const;
    double getMaxFreqValue() const;
    bool getEnabled() const;
    double yMax() const;
    OverlayType type() const { return d_type; }
    QString errorString() const { return d_errorString; }
    bool isModified() const { return d_modified; }
    bool isPreview() const { return d_preview; }
    
    // Curve metadata access methods
    QVariant getCurveMetadata(const QString &key) const;
    void setCurveMetadata(const QString &key, const QVariant &value);
    void setCurveAppearanceMetadata(const BC::Data::CurveAppearance &appearance);
    
    void setLabel(const QString &newlabel);
    void setSourceFile(const QString &newsourceFile);
    void setDestFile(const QString &newdestFile);
    void setPlotId(const QString &newplotId);
    void setComment(const QString &newcomment);
    void setYScale(double newyScale);
    void setYOffset(double newyOffset);
    void setXOffset(double newxOffset);
    void setMinFreqLimit(bool enabled, double value);
    void setMaxFreqLimit(bool enabled, double value);
    void setEnabled(bool enabled);
    void setPreview(bool preview);
    
    void save();

    
    
protected:
    void setModified(bool modified = true) { d_modified = modified; }
    
    virtual void readFromDest() =0;
    virtual void writeToDest() =0;
    
    virtual void _storeMetadata(std::map<QString,QVariant> &m) =0;
    virtual void _retrieveMetadata(const std::map<QString,QVariant> &m) =0;
    
    void invalidateCache();

    QString d_errorString;
    
private:
    virtual QVector<QPointF> _xyData() const = 0;
    
    OverlayType d_type;
    QString d_label{""}, d_sourceFile{""}, d_destFile{""}, d_plotId{""}, d_comment{""};
    double d_yScale{1.0}, d_yOffset{0.0}, d_xOffset{0.0};
    
    // Frequency range filtering
    bool d_minFreqEnabled{false}, d_maxFreqEnabled{false};
    double d_minFreqValue{0.0}, d_maxFreqValue{1000.0};
    
    // Overlay visibility control
    bool d_enabled{true};
    
    // Preview mode control (prevents disk writing)
    bool d_preview{false};
    
    bool d_modified{false};
    
    // Curve metadata storage for direct access by OverlayMetadataStorage
    std::map<QString, QVariant> d_curveMetadata;
    
    // Caching for filtered xyData
    mutable QVector<QPointF> d_cachedFilteredData;
    mutable bool d_cacheValid{false};
    mutable double d_cachedYMax{0.0};
    
    
    void storeMetadata(std::map<QString,QVariant> &m);
    void retrieveMetadata(const std::map<QString,QVariant> &m);
};

#endif // OVERLAYBASE_H
