#ifndef OVERLAYBASE_H
#define OVERLAYBASE_H

#include <map>
#include <QString>
#include <QPointF>

class OverlayBase
{
public:
    OverlayBase(int num, QString path = "");
    
    virtual QVector<QPointF> xyData() const =0;
    
    QString getLabel() const;
    QString getSourceFile() const;
    QString getDestFile() const;
    QString getPlotId() const;
    double getYScale() const;
    double getYOffset() const;
    double getXOffset() const;
    
    void setLabel(const QString &newlabel);
    void setSourceFile(const QString &newsourceFile);
    void setDestFile(const QString &newdestFile);
    void setPlotId(const QString &newplotId);
    void setYScale(double newyScale);
    void setYOffset(double newyOffset);
    void setXOffset(double newxOffset);
    
    void loadFromSource();
    
    void writeMetadata(std::map<QString,QVariant> &m);
    void readMetadata(const std::map<QString,QVariant> &m);
    
    
protected:
    virtual void readFromSource() =0;
    virtual void readFromDest() =0;
    virtual void writeToDest() =0;
    
    virtual void _writeMetadata(std::map<QString,QVariant> &m) =0;
    virtual void _readMetadata(const std::map<QString,QVariant> &m) =0;
    
private:
    int d_number;
    QString d_path;
    
    QString d_label, d_sourceFile, d_destFile, d_plotId;
    double d_yScale{1.0}, d_yOffset{0.0}, d_xOffset{0.0};
    
    bool d_modified{false};
};

#endif // OVERLAYBASE_H
