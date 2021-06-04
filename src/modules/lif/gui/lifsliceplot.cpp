#include "lifsliceplot.h"

#include <QSettings>
#include <QPalette>
#include <QFileDialog>
#include <QMessageBox>
#include <QMenu>

#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_textlabel.h>
#include <qwt6/qwt_symbol.h>
#include <qwt6/qwt_scale_widget.h>

#include <src/data/datastructs.h>

LifSlicePlot::LifSlicePlot(QWidget *parent) :
    ZoomPanPlot(QString("lifSlicePlot"),parent)
{
    setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));

    QwtText llabel(QString("LIF (AU)"));
    llabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::yLeft,llabel);

    p_curve = new QwtPlotCurve(QString("trace"));
    p_curve->setRenderHint(QwtPlotItem::RenderAntialiased);
    auto pal = QPalette();
    p_curve->setSymbol(new QwtSymbol(QwtSymbol::Ellipse,palette().text(),QPen(QPalette::Text),QSize(8,8)));
    p_curve->setZ(1.0);
    p_curve->attach(this);

}

LifSlicePlot::~LifSlicePlot()
{

}

void LifSlicePlot::setXAxisTitle(QString title)
{
    QwtText label(title);
    label.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::xBottom,label);

    replot();
}

void LifSlicePlot::setName(QString name)
{
    ZoomPanPlot::setName(name);

    QSettings s;
    QColor c = s.value(QString("%1/curveColor").arg(d_name),QPalette().color(QPalette::Text)).value<QColor>();
    p_curve->setPen(QPen(c));
}

void LifSlicePlot::prepareForExperiment(double xMin, double xMax)
{
    p_curve->setSamples(QVector<QPointF>());
    d_currentData = QVector<QPointF>();

    setAxisAutoScaleRange(QwtPlot::xBottom,xMin,xMax);
    setAxisAutoScaleRange(QwtPlot::yLeft,0.0,1.0);

    autoScale();
}

void LifSlicePlot::setData(const QVector<QPointF> d)
{
    d_currentData = d;
    filterData();
    replot();
}

void LifSlicePlot::setPlotTitle(QString text)
{
    QFont f(QString("sans-serif"),8);
    QwtText title(text);
    title.setFont(f);
    setTitle(title);
}

void LifSlicePlot::exportXY()
{
    QString path = BlackChirp::getExportDir();
    QString name = QFileDialog::getSaveFileName(this,QString("Export LIF Trace"),path + QString("/lifslice.txt"));
    if(name.isEmpty())
        return;
    QFile f(name);
    if(!f.open(QIODevice::WriteOnly))
    {
        QMessageBox::critical(this,QString("Export Failed"),QString("Could not open file %1 for writing. Please choose a different filename.").arg(name));
        return;
    }
    QApplication::setOverrideCursor(Qt::BusyCursor);
    QString xlabel = this->axisWidget(QwtPlot::xBottom)->title().text();
    f.write(QString("%1\tlif_integral").arg(xlabel).toLatin1());
    auto d = d_currentData;
    for(int i=0;i<d.size();i++)
    {
        f.write(QString("\n%1\t%2").arg(d.at(i).x(),0,'e',6)
                    .arg(d.at(i).y(),0,'e',12).toLatin1());
    }
    f.close();
    QApplication::restoreOverrideCursor();
    BlackChirp::setExportDir(name);
}

void LifSlicePlot::filterData()
{
    if(d_currentData.size() < 2)
    {
        p_curve->setSamples(QVector<QPointF>());
        return;
    }

    double firstPixel = 0.0;
    double lastPixel = canvas()->width();
    QwtScaleMap map = canvasMap(QwtPlot::xBottom);

    QVector<QPointF> filtered;
    filtered.reserve(canvas()->width()*2);

    //find first data point that is in the range of the plot
    int dataIndex = 0;
    while(dataIndex+1 < d_currentData.size() && map.transform(d_currentData.at(dataIndex).x()) < firstPixel)
        dataIndex++;

    //add the previous point to the filtered array
    //this will make sure the curve always goes to the edge of the plot
    if(dataIndex-1 >= 0)
        filtered.append(d_currentData.at(dataIndex-1));

    //at this point, dataIndex is at the first point within the range of the plot. loop over pixels, compressing data
    for(double pixel = firstPixel; pixel<lastPixel; pixel+=1.0)
    {
        double min = d_currentData.at(dataIndex).y(), max = min;
        int numPnts = 0;
        double nextPixelX = map.invTransform(pixel+1.0);
        while(dataIndex+1 < d_currentData.size() && d_currentData.at(dataIndex).x() < nextPixelX)
        {
            auto pt = d_currentData.at(dataIndex);
            min = qMin(pt.y(),min);
            max = qMax(pt.y(),max);

            dataIndex++;
            numPnts++;
        }



        if(numPnts == 1)
            filtered.append(d_currentData.at(dataIndex-1));
        else if (numPnts > 1)
        {
            QPointF first(map.invTransform(pixel),min);
            QPointF second(map.invTransform(pixel),max);
            filtered.append(first);
            filtered.append(second);
        }
    }

    if(dataIndex < d_currentData.size())
        filtered.append(d_currentData.at(dataIndex));


    //assign data to curve object
    p_curve->setSamples(filtered);

}


QMenu *LifSlicePlot::contextMenu()
{
    QMenu *out = ZoomPanPlot::contextMenu();

    out->addAction(QString("Export XY..."),this,&LifSlicePlot::exportXY);
    if(d_currentData.size()<2)
        out->setEnabled(false);

    return out;
}
