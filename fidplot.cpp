#include "fidplot.h"
#include <math.h>

#include <QPalette>
#include <QSettings>
#include <QApplication>
#include <QMenu>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QWidgetAction>
#include <QFormLayout>
#include <QColorDialog>
#include <QMouseEvent>
#include <QFileDialog>
#include <QMessageBox>

#include <qwt6/qwt_plot_canvas.h>
#include <qwt6/qwt_plot_marker.h>
#include <qwt6/qwt_plot_curve.h>

FidPlot::FidPlot(QWidget *parent) :
    ZoomPanPlot(QString("FidPlot"),parent), d_ftEndAtFidEnd(true), d_number(0)
{
    //make axis label font smaller
    this->setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    this->setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));

    //build axis titles with small font. The <html> etc. tags are needed to display the mu character
    QwtText blabel(QString("<html><body>Time (&mu;s)</body></html>"));
    blabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::xBottom,blabel);

    QwtText llabel(QString("FID"));
    llabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::yLeft,llabel);

//    this->setAxisScaleDraw(QwtPlot::yLeft,new SciNotationScaleDraw());


    p_curve = new QwtPlotCurve();
    QSettings s;
    QColor c = s.value(QString("fidcolor"),palette().color(QPalette::Text)).value<QColor>();
    p_curve->setPen(QPen(c));
    p_curve->setRenderHint(QwtPlotItem::RenderAntialiased);
    p_curve->attach(this);
    p_curve->setVisible(false);

    QwtPlotMarker *chirpStartMarker = new QwtPlotMarker();
    chirpStartMarker->setLineStyle(QwtPlotMarker::VLine);
    chirpStartMarker->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText csLabel(QString("Chirp Start"));
    csLabel.setFont(QFont(QString("sans serif"),6));
    chirpStartMarker->setLabel(csLabel);
    chirpStartMarker->setLabelOrientation(Qt::Vertical);
    chirpStartMarker->setLabelAlignment(Qt::AlignBottom|Qt::AlignRight);
    d_chirpMarkers.first = chirpStartMarker;
    chirpStartMarker->attach(this);
    chirpStartMarker->setVisible(false);

    QwtPlotMarker *chirpEndMarker = new QwtPlotMarker();
    chirpEndMarker->setLineStyle(QwtPlotMarker::VLine);
    chirpEndMarker->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText ceLabel(QString("Chirp End"));
    ceLabel.setFont(QFont(QString("sans serif"),6));
    chirpEndMarker->setLabel(ceLabel);
    chirpEndMarker->setLabelOrientation(Qt::Vertical);
    chirpEndMarker->setLabelAlignment(Qt::AlignTop|Qt::AlignRight);
    d_chirpMarkers.second = chirpEndMarker;
    chirpEndMarker->attach(this);
    chirpEndMarker->setVisible(false);

    QSettings s2(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s2.beginGroup(QString("FidPlot"));
    double ftStart = s2.value(QString("lastFtStart"),0.0).toDouble();
    double ftEnd = s2.value(QString("lastFtEnd"),-1.0).toDouble();
    s2.endGroup();

    QwtPlotMarker *ftStartMarker = new QwtPlotMarker();
    ftStartMarker->setLineStyle(QwtPlotMarker::VLine);
    ftStartMarker->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText ftsLabel(QString(" FT Start "));
    ftsLabel.setFont(QFont(QString("sans serif"),6));
    ftsLabel.setBackgroundBrush(QPalette().window());
    ftsLabel.setColor(QPalette().text().color());
    ftStartMarker->setLabel(ftsLabel);
    ftStartMarker->setLabelOrientation(Qt::Vertical);
    ftStartMarker->setLabelAlignment(Qt::AlignBottom|Qt::AlignRight);
    ftStartMarker->setXValue(ftStart);
    ftStartMarker->attach(this);
    ftStartMarker->setVisible(false);
    d_ftMarkers.first = ftStartMarker;

    QwtPlotMarker *ftEndMarker = new QwtPlotMarker();
    ftEndMarker->setLineStyle(QwtPlotMarker::VLine);
    ftEndMarker->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText fteLabel(QString(" FT End "));
    fteLabel.setFont(QFont(QString("sans serif"),6));
    fteLabel.setBackgroundBrush(QPalette().window());
    fteLabel.setColor(QPalette().text().color());
    ftEndMarker->setLabel(fteLabel);
    ftEndMarker->setLabelOrientation(Qt::Vertical);
    ftEndMarker->setLabelAlignment(Qt::AlignTop|Qt::AlignLeft);
    ftEndMarker->setXValue(ftEnd);
    ftEndMarker->attach(this);
    ftEndMarker->setVisible(false);
    d_ftMarkers.second = ftEndMarker;

    connect(this,&FidPlot::plotRightClicked,this,&FidPlot::buildContextMenu);

    setAxisAutoScaleRange(QwtPlot::yLeft,0.0,1.0);

}

void FidPlot::receiveData(const Fid f)
{
    if(f.size() < 2)
        return;

    d_currentFid = f;

    filterData();
    replot();
}

void FidPlot::filterData()
{
    if(d_currentFid.size() < 2)
        return;

    QVector<QPointF> fidData = d_currentFid.toXY();

    double firstPixel = 0.0;
    double lastPixel = canvas()->width();
    QwtScaleMap map = canvasMap(QwtPlot::xBottom);

    QVector<QPointF> filtered;

    //find first data point that is in the range of the plot
    int dataIndex = 0;
    while(dataIndex+1 < fidData.size() && map.transform(fidData.at(dataIndex).x()*1e6) < firstPixel)
        dataIndex++;

    //add the previous point to the filtered array
    //this will make sure the curve always goes to the edge of the plot
    if(dataIndex-1 >= 0)
        filtered.append(fidData.at(dataIndex-1));

    //at this point, dataIndex is at the first point within the range of the plot. loop over pixels, compressing data
    double yMin = fidData.at(dataIndex).y(), yMax = yMin;
    for(double pixel = firstPixel; pixel<lastPixel; pixel+=1.0)
    {
        double min = fidData.at(dataIndex).y(), max = min;
        int minIndex = dataIndex, maxIndex = dataIndex;
        int numPnts = 0;
        while(dataIndex+1 < fidData.size() && map.transform(fidData.at(dataIndex).x()*1e6) < pixel+1.0)
        {
            if(fidData.at(dataIndex).y() < min)
            {
                min = fidData.at(dataIndex).y();
                minIndex = dataIndex;
            }
            if(fidData.at(dataIndex).y() > max)
            {
                max = fidData.at(dataIndex).y();
                maxIndex = dataIndex;
            }
            dataIndex++;
            numPnts++;
        }
        if(filtered.isEmpty())
        {
            yMin = min;
            yMax = max;
        }
        else
        {
            yMin = qMin(min,yMin);
            yMax = qMax(max,yMax);
        }
        if(numPnts == 1)
            filtered.append(QPointF(fidData.at(dataIndex-1).x()*1e6,fidData.at(dataIndex-1).y()));
        else if (numPnts > 1)
        {
            QPointF first(map.invTransform(pixel),fidData.at(minIndex).y());
            QPointF second(map.invTransform(pixel),fidData.at(maxIndex).y());
            filtered.append(first);
            filtered.append(second);
        }
    }

    if(dataIndex < fidData.size())
    {
        QPointF p = fidData.at(dataIndex);
        p.setX(p.x()*1e6);
        filtered.append(p);
    }

    expandAutoScaleRange(QwtPlot::yLeft,yMin,yMax);
    //assign data to curve object
    p_curve->setSamples(filtered);
}

void FidPlot::prepareForExperiment(const Experiment e)
{     
    FtmwConfig c = e.ftmwConfig();
    d_number = e.number();
    d_currentFid = Fid();
    p_curve->setSamples(QVector<QPointF>());

    if(!c.isEnabled())
    {
        p_curve->setVisible(false);

        d_chirpMarkers.first->setVisible(false);
        d_chirpMarkers.second->setVisible(false);
        d_ftMarkers.first->setVisible(false);
        d_ftMarkers.second->setVisible(false);

        setAxisAutoScaleRange(QwtPlot::xBottom,0.0,1.0);
        setAxisAutoScaleRange(QwtPlot::yLeft,0.0,1.0);
    }
    else
    {
        p_curve->setVisible(true);

        d_ftMarkers.first->setVisible(true);
        d_ftMarkers.second->setVisible(true);

        double maxTime = c.scopeConfig().recordLength/c.scopeConfig().sampleRate*1e6;
        double ftEnd = d_ftMarkers.second->xValue();
        if(ftEnd < 0.0 || ftEnd < d_ftMarkers.first->xValue() || ftEnd > maxTime)
            d_ftMarkers.second->setXValue(maxTime);

        emit ftStartChanged(d_ftMarkers.first->xValue());
        emit ftEndChanged(d_ftMarkers.second->xValue());

        setAxisAutoScaleRange(QwtPlot::xBottom,0.0,maxTime);
        setAxisAutoScaleRange(QwtPlot::yLeft,0.0,0.0);

        //following will only matter if phase correction is even implemented
        bool displayMarkers = c.isPhaseCorrectionEnabled();
        if(displayMarkers)
        {
            double chirpStart = c.chirpConfig().preChirpDelay() + c.chirpConfig().preChirpProtection() - c.scopeConfig().trigDelay*1e6;
            double chirpEnd = chirpStart + c.chirpConfig().chirpDuration();

            d_chirpMarkers.first->setValue(chirpStart,0.0);
            d_chirpMarkers.second->setValue(chirpEnd,0.0);
        }

        d_chirpMarkers.first->setVisible(displayMarkers);
        d_chirpMarkers.second->setVisible(displayMarkers);
    }

    autoScale();
}

void FidPlot::setFtStart(double start)
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("FidPlot"));

    if(start < d_ftMarkers.second->value().x() && start >= 0.0)
    {
        d_ftMarkers.first->setValue(start,0.0);
        emit ftStartChanged(start);
    }
    else
        emit overrideStart(d_ftMarkers.first->value().x());

    s.setValue(QString("lastFtStart"),d_ftMarkers.first->xValue());
    s.endGroup();

    QwtPlot::replot();
}

void FidPlot::setFtEnd(double end)
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("FidPlot"));

    if(end > d_ftMarkers.first->value().x() && end <= d_currentFid.spacing()*d_currentFid.size()*1e6)
    {
        d_ftMarkers.second->setValue(end,0.0);
        emit ftEndChanged(end);
    }
    else
        emit overrideEnd(d_ftMarkers.second->value().x());

    s.setValue(QString("lastFtEnd"),d_ftMarkers.second->xValue());
    s.endGroup();

    QwtPlot::replot();
}

void FidPlot::buildContextMenu(QMouseEvent *me)
{
    if(d_currentFid.size()<2 || !isEnabled())
        return;

    QMenu *menu = contextMenu();

    QAction *colorAct = menu->addAction(QString("Change FID color..."));
    connect(colorAct,&QAction::triggered,this,&FidPlot::changeFidColor);

    QAction *exportAct = menu->addAction(QString("Export to ASCII..."));
    if(d_currentFid.size() == 0)
        exportAct->setEnabled(false);
    connect(exportAct,&QAction::triggered,this,&FidPlot::exportFid);

    QWidgetAction *wa = new QWidgetAction(menu);
    QWidget *w = new QWidget(menu);
    QFormLayout *fl = new QFormLayout(w);



    QDoubleSpinBox *startBox = new QDoubleSpinBox();
    startBox->setMinimum(0.0);
    startBox->setDecimals(2);
    startBox->setMaximum(d_currentFid.size()*d_currentFid.spacing()*1e6);
    startBox->setSingleStep(0.05);
    startBox->setValue(d_ftMarkers.first->value().x());
    startBox->setKeyboardTracking(false);
    connect(startBox,static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),this,&FidPlot::setFtStart);
    connect(this,&FidPlot::overrideStart,startBox,&QDoubleSpinBox::setValue);
    startBox->setSuffix(QString::fromUtf8(" μs"));
    fl->addRow(QString("FT Start"),startBox);

    QDoubleSpinBox *endBox = new QDoubleSpinBox();
    endBox->setMinimum(0.0);
    endBox->setDecimals(2);
    endBox->setMaximum(d_currentFid.size()*d_currentFid.spacing()*1e6);
    endBox->setSingleStep(0.05);
    endBox->setValue(d_ftMarkers.second->value().x());
    endBox->setKeyboardTracking(false);
    connect(endBox,static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),this,&FidPlot::setFtEnd);
    connect(this,&FidPlot::overrideEnd,endBox,&QDoubleSpinBox::setValue);
    endBox->setSuffix(QString::fromUtf8(" μs"));
    fl->addRow(QString("FT End"),endBox);

    w->setLayout(fl);
    wa->setDefaultWidget(w);
    menu->addAction(wa);

    menu->popup(me->globalPos());

}

void FidPlot::changeFidColor()
{
    QColor c = QColorDialog::getColor(p_curve->pen().color(),this,QString("Select Color"));
    if(c.isValid())
    {
        p_curve->setPen(c);

        QSettings s;
        s.setValue(QString("fidcolor"),c);
        s.sync();

        replot();
    }
}

void FidPlot::exportFid()
{
    QString name = QFileDialog::getSaveFileName(this,QString("Export FID"),QString("~/%1_fid.txt").arg(d_number));
    if(name.isEmpty())
        return;

    QFile f(name);

    if(!f.open(QIODevice::WriteOnly))
    {
        QMessageBox::critical(this,QString("FID Export Failed"),QString("Could not open file %1 for writing. Please choose a different filename.").arg(name));
        return;
    }

    QApplication::setOverrideCursor(Qt::BusyCursor);

    f.write(QString("#Probe freq\t%1\tMHz").arg(d_currentFid.probeFreq(),0,'f',5).toLatin1());
    if(d_currentFid.sideband() == BlackChirp::UpperSideband)
        f.write(QString("\n#Sideband\tUpper\t").toLatin1());
    else
        f.write(QString("\n#Sideband\tLower\t").toLatin1());
    f.write(QString("\n#Shots\t%1\t").arg(d_currentFid.shots()).toLatin1());
    f.write(QString("\n#Spacing\t%1\ts\n\n").arg(d_currentFid.spacing(),0,'e',3).toLatin1());

    f.write(QString("fid%1").arg(d_number).toLatin1());

    for(int i=0;i<d_currentFid.size();i++)
        f.write(QString("\n%1").arg(d_currentFid.at(i),0,'e',12).toLatin1());
    f.close();

    QApplication::restoreOverrideCursor();
}
