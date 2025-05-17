#ifndef OVERLAY_H
#define OVERLAY_H

#include <QString>
#include <QPointF>

class OverlayBase
{
public:
    OverlayBase(int num, QString path = "");
    
    virtual QVector<QPointF> xyData() =0;
    
private:
    int d_number;
    QString d_path;
    
    QString d_label, d_sourceFile, d_destFile, d_plotId;
    double d_yScale{1.0}, d_yOffset{0.0}, d_xOffset{0.0};
    
    bool d_modified{false};
};

#endif // OVERLAY_H
