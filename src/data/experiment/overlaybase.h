#ifndef OVERLAYBASE_H
#define OVERLAYBASE_H

#include <map>
#include <QString>
#include <QPointF>
#include <QVariant>

class OverlayMetadataStorage;

namespace BC::Key::Overlay {
static const QString oLabel{"label"};
static const QString oSourceFile{"sourceFile"};
static const QString oDestFile{"destFile"};
static const QString oPlotId{"plotId"};
static const QString oYScale{"yScale"};
static const QString oYOffset{"yOffset"};
static const QString oXOffset{"xOffset"};
static const QString overlaySettingsFile{"%1.settings.csv"};
static const QString overlayDataFile{"%1.data.csv"};
}

class OverlayBase
{
    Q_GADGET
    friend class OverlayStorage;
    friend class OverlayMetadataStorage;
    
public:
    enum OverlayType {
        BCExperiment,
        SPCAT,
        GenericXY
    };
    Q_ENUM(OverlayType)

    OverlayBase(OverlayType type);
    
    
    virtual QVector<QPointF> xyData() const =0;
    
    QString getLabel() const;
    QString getSourceFile() const;
    QString getDestFile() const;
    QString getPlotId() const;
    double getYScale() const;
    double getYOffset() const;
    double getXOffset() const;
    OverlayType type() const { return d_type; }
    QString errorString() const { return d_errorString; }
    bool isModified() const { return d_modified; }
    
    void setLabel(const QString &newlabel);
    void setSourceFile(const QString &newsourceFile);
    void setDestFile(const QString &newdestFile);
    void setPlotId(const QString &newplotId);
    void setYScale(double newyScale);
    void setYOffset(double newyOffset);
    void setXOffset(double newxOffset);
    
    void save();

    
    
protected:
    void setModified(bool modified = true) { d_modified = modified; }
    
    virtual void readFromDest() =0;
    virtual void writeToDest() =0;
    
    virtual void _storeMetadata(std::map<QString,QVariant> &m) =0;
    virtual void _retrieveMetadata(const std::map<QString,QVariant> &m) =0;

    QString d_errorString;
    
private:
    OverlayType d_type;
    QString d_label, d_sourceFile, d_destFile, d_plotId;
    double d_yScale{1.0}, d_yOffset{0.0}, d_xOffset{0.0};
    
    bool d_modified{false};
    
    
    void storeMetadata(std::map<QString,QVariant> &m);
    void retrieveMetadata(const std::map<QString,QVariant> &m);
};

#endif // OVERLAYBASE_H
