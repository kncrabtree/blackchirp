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
#include <src/gui/plot/blackchirpplotcurve.h>

LifSlicePlot::LifSlicePlot(const QString name, QWidget *parent) :
    ZoomPanPlot(name,parent)
{
    this->setPlotAxisTitle(QwtPlot::yLeft,QString("LIF (AU)"));

    p_curve = new BlackchirpPlotCurve(BC::Key::lifSliceCurve,Qt::SolidLine,QwtSymbol::Ellipse);
    p_curve->setZ(1.0);
    p_curve->attach(this);

}

LifSlicePlot::~LifSlicePlot()
{

}

void LifSlicePlot::prepareForExperiment()
{
    p_curve->setCurveData(QVector<QPointF>());
    autoScale();
}

void LifSlicePlot::setData(const QVector<QPointF> d)
{
    p_curve->setCurveData(d);
    replot();
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
    auto d = p_curve->curveData();
    for(int i=0;i<d.size();i++)
    {
        f.write(QString("\n%1\t%2").arg(d.at(i).x(),0,'e',6)
                    .arg(d.at(i).y(),0,'e',12).toLatin1());
    }
    f.close();
    QApplication::restoreOverrideCursor();
    BlackChirp::setExportDir(name);
}
