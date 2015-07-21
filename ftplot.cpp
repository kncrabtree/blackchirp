#include "ftplot.h"

#include <QFont>
#include <QMouseEvent>
#include <QEvent>
#include <QSettings>
#include <QMenu>
#include <QAction>
#include <QActionGroup>
#include <QColorDialog>
#include <QApplication>
#include <QWidgetAction>
#include <QSpinBox>
#include <QFormLayout>
#include <QFileDialog>
#include <QMessageBox>

#include <qwt6/qwt_picker_machine.h>
#include <qwt6/qwt_scale_widget.h>
#include <qwt6/qwt_scale_div.h>
#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_picker.h>
#include <qwt6/qwt_plot_grid.h>

FtPlot::FtPlot(QWidget *parent) :
    ZoomPanPlot(QString("FtPlot"),parent), d_number(0), d_pzf(0)
{
    //make axis label font smaller
    this->setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    this->setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));

    //build axis titles with small font
    QwtText blabel(QString("Frequency (MHz)"));
    blabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::xBottom,blabel);

    QwtText llabel(QString("FT"));
    llabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::yLeft,llabel);

    QSettings s;

    //build and configure curve object
    p_curveData = new QwtPlotCurve();
    QColor c = s.value(QString("ftcolor"),palette().color(QPalette::BrightText)).value<QColor>();
    p_curveData->setPen(QPen(c));
    p_curveData->setRenderHint(QwtPlotItem::RenderAntialiased);
    p_curveData->attach(this);
    p_curveData->setVisible(false);

    QwtPlotPicker *picker = new QwtPlotPicker(this->canvas());
    picker->setAxis(QwtPlot::xBottom,QwtPlot::yLeft);
    picker->setStateMachine(new QwtPickerClickPointMachine);
    picker->setMousePattern(QwtEventPattern::MouseSelect1,Qt::RightButton);
    picker->setTrackerMode(QwtPicker::AlwaysOn);
    picker->setTrackerPen(QPen(QPalette().color(QPalette::Text)));
    picker->setEnabled(true);

    p_plotGrid = new QwtPlotGrid();
    p_plotGrid->enableX(true);
    p_plotGrid->enableXMin(true);
    p_plotGrid->enableY(true);
    p_plotGrid->enableYMin(true);
    QPen p;
    p.setColor(s.value(QString("gridcolor"),palette().color(QPalette::Light)).value<QColor>());
    p.setStyle(Qt::DashLine);
    p_plotGrid->setMajorPen(p);
    p.setStyle(Qt::DotLine);
    p_plotGrid->setMinorPen(p);
    p_plotGrid->attach(this);

    connect(this,&FtPlot::plotRightClicked,this,&FtPlot::buildContextMenu);

    setAxisAutoScaleRange(QwtPlot::xBottom,0.0,1.0);
    setAxisAutoScaleRange(QwtPlot::yLeft,0.0,1.0);

}

FtPlot::~FtPlot()
{
}

void FtPlot::prepareForExperiment(const Experiment e)
{
    FtmwConfig c = e.ftmwConfig();
    d_number = e.number();

    d_currentFt = QVector<QPointF>();
    p_curveData->setSamples(d_currentFt);
    setAxisAutoScaleRange(QwtPlot::yLeft,0.0,1.0);
    if(!c.isEnabled())
    {
        p_curveData->setVisible(false);
        setAxisAutoScaleRange(QwtPlot::xBottom,0.0,1.0);
    }
    else
    {
        p_curveData->setVisible(true);
        setAxisAutoScaleRange(QwtPlot::xBottom,c.ftMin(),c.ftMax());
    }
    autoScale();
}

void FtPlot::newFt(QVector<QPointF> ft, double max)
{
    d_currentFt = ft;
    if(ft.isEmpty())
        return;

    setAxisAutoScaleMax(QwtPlot::yLeft,max);
    filterData();
    replot();
}

void FtPlot::filterData()
{
    if(d_currentFt.size() < 2)
        return;

    double firstPixel = 0.0;
    double lastPixel = canvas()->width();
    QwtScaleMap map = canvasMap(QwtPlot::xBottom);

    QVector<QPointF> filtered;
    filtered.reserve(canvas()->width()*2);

    //find first data point that is in the range of the plot
    int dataIndex = 0;
    while(dataIndex+1 < d_currentFt.size() && map.transform(d_currentFt.at(dataIndex).x()) < firstPixel)
        dataIndex++;

    //add the previous point to the filtered array
    //this will make sure the curve always goes to the edge of the plot
    if(dataIndex-1 >= 0)
        filtered.append(d_currentFt.at(dataIndex-1));

    //at this point, dataIndex is at the first point within the range of the plot. loop over pixels, compressing data
    for(double pixel = firstPixel; pixel<lastPixel; pixel+=1.0)
    {
        double min = d_currentFt.at(dataIndex).y(), max = min;
        int minIndex = dataIndex, maxIndex = dataIndex;
        int numPnts = 0;
        while(dataIndex+1 < d_currentFt.size() && map.transform(d_currentFt.at(dataIndex).x()) < pixel+1.0)
        {
            if(d_currentFt.at(dataIndex).y() < min)
            {
                min = d_currentFt.at(dataIndex).y();
                minIndex = dataIndex;
            }
            if(d_currentFt.at(dataIndex).y() > max)
            {
                max = d_currentFt.at(dataIndex).y();
                maxIndex = dataIndex;
            }
            dataIndex++;
            numPnts++;
        }

        if(numPnts == 1)
            filtered.append(d_currentFt.at(dataIndex-1));
        else if (numPnts > 1)
        {
            QPointF first(map.invTransform(pixel),d_currentFt.at(minIndex).y());
            QPointF second(map.invTransform(pixel),d_currentFt.at(maxIndex).y());
            filtered.append(first);
            filtered.append(second);
        }
    }

    if(dataIndex < d_currentFt.size())
        filtered.append(d_currentFt.at(dataIndex));

    //assign data to curve object
    p_curveData->setSamples(filtered);
}

void FtPlot::buildContextMenu(QMouseEvent *me)
{
    if(d_currentFt.size() < 2 || !isEnabled())
        return;

    QMenu *m = contextMenu();

    QAction *ftColorAction = m->addAction(QString("Change FT Color..."));
    connect(ftColorAction,&QAction::triggered,this,[=](){ changeFtColor(getColor(p_curveData->pen().color())); });

    QAction *gridColorAction = m->addAction(QString("Change Grid Color..."));
    connect(gridColorAction,&QAction::triggered,this,[=](){ changeGridColor(getColor(p_plotGrid->majorPen().color())); });

    QAction *exportAction = m->addAction(QString("Export XY..."));
    connect(exportAction,&QAction::triggered,this,&FtPlot::exportXY);

    QWidgetAction *wa = new QWidgetAction(m);
    QWidget *w = new QWidget(m);
    QSpinBox *pzfBox = new QSpinBox(w);
    QFormLayout *fl = new QFormLayout();

    fl->addRow(QString("Zero fill factor"),pzfBox);

    pzfBox->setRange(0,4);
    pzfBox->setSingleStep(1);
    pzfBox->setValue(d_pzf);
    connect(pzfBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),[=](int p){
        d_pzf = p;
        emit pzfChanged(p);
    });

    w->setLayout(fl);
    wa->setDefaultWidget(w);
    m->addAction(wa);

    m->popup(me->globalPos());
}

void FtPlot::changeFtColor(QColor c)
{
    if(!c.isValid())
        return;

    QSettings s;
    s.setValue(QString("ftcolor"),c);
    s.sync();

    p_curveData->setPen(QPen(c));
    replot();

}

void FtPlot::changeGridColor(QColor c)
{
    if(!c.isValid())
        return;

    QSettings s;
    s.setValue(QString("gridcolor"),c);
    s.sync();

    QPen p(c);
    p.setStyle(Qt::DashLine);
    p_plotGrid->setMajorPen(p);

    p.setStyle(Qt::DotLine);
    p_plotGrid->setMinorPen(p);
    replot();
}

QColor FtPlot::getColor(QColor startingColor)
{
    return QColorDialog::getColor(startingColor,this,QString("Select Color"));
}

void FtPlot::exportXY()
{
    QString name = QFileDialog::getSaveFileName(this,QString("Export FT"),QString("~/%1_ft.txt").arg(d_number));
    if(name.isEmpty())
        return;

    QFile f(name);

    if(!f.open(QIODevice::WriteOnly))
    {
        QMessageBox::critical(this,QString("FT Export Failed"),QString("Could not open file %1 for writing. Please choose a different filename.").arg(name));
        return;
    }

    QApplication::setOverrideCursor(Qt::BusyCursor);

    f.write(QString("freq%1\tft%1").arg(d_number).toLatin1());

    for(int i=0;i<d_currentFt.size();i++)
        f.write(QString("\n%1\t%2").arg(d_currentFt.at(i).x(),0,'f',6).arg(d_currentFt.at(i).y(),0,'e',12).toLatin1());
    f.close();

    QApplication::restoreOverrideCursor();
}
