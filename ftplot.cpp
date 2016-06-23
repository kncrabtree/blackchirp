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
#include <QList>
#include <QPair>

#include <qwt6/qwt_picker_machine.h>
#include <qwt6/qwt_scale_widget.h>
#include <qwt6/qwt_scale_div.h>
#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_picker.h>
#include <qwt6/qwt_plot_grid.h>

FtPlot::FtPlot(QWidget *parent) :
    ZoomPanPlot(QString("FtPlot"),parent), d_number(0), d_pzf(0), d_currentUnits(BlackChirp::FtPlotmV)
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

    d_currentWinf = static_cast<BlackChirp::FtWindowFunction>(s.value(QString("windowFunction"),BlackChirp::Boxcar).toInt());

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
    QSettings s;
    configureUnits(static_cast<BlackChirp::FtPlotUnits>(s.value(QString("ftUnits"),BlackChirp::FtPlotmV).toInt()));
    setWinf(d_currentWinf);
}

void FtPlot::newFt(QVector<QPointF> ft, double max)
{
    d_currentFt = ft;
    if(ft.isEmpty())
        return;

    setAxisAutoScaleRange(QwtPlot::yLeft,0.0,max);
    filterData();
    replot();
}

void FtPlot::newFtDiff(const QVector<QPointF> ft, double min, double max)
{
    d_currentFt = ft;
    if(ft.isEmpty())
        return;

    setAxisAutoScaleRange(QwtPlot::yLeft,min,max);
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

    QList<QPair<BlackChirp::FtPlotUnits,QString>> unitsList;
    unitsList << qMakePair(BlackChirp::FtPlotV,QString("V"));
    unitsList << qMakePair(BlackChirp::FtPlotmV,QString("mV"));
    unitsList << qMakePair(BlackChirp::FtPlotuV,QString::fromUtf16(u"µV"));
    unitsList << qMakePair(BlackChirp::FtPlotnV,QString("nV"));


    QMenu *yMenu = m->addMenu(QString("Y Scaling"));
    QActionGroup *scaleGroup = new QActionGroup(yMenu);
    scaleGroup->setExclusive(true);

    for(int i=0; i<unitsList.size(); i++)
    {
        QAction *a = yMenu->addAction(unitsList.at(i).second);
        a->setCheckable(true);
        if(unitsList.at(i).first == d_currentUnits)
            a->setChecked(true);
        else
            a->setChecked(false);
        connect(a,&QAction::triggered,this,[=](){ configureUnits(unitsList.at(i).first); });
    }
    yMenu->addActions(scaleGroup->actions());

    QList<QPair<BlackChirp::FtWindowFunction,QString>> winfList;
    winfList << qMakePair(BlackChirp::Bartlett,QString("Bartlett"));
    winfList << qMakePair(BlackChirp::Blackman,QString("Blackman"));
    winfList << qMakePair(BlackChirp::BlackmanHarris,QString("Blackman-Harris"));
    winfList << qMakePair(BlackChirp::Boxcar,QString("Boxcar (none)"));
    winfList << qMakePair(BlackChirp::Hamming,QString("Hamming"));
    winfList << qMakePair(BlackChirp::Hanning,QString("Hanning"));
    winfList << qMakePair(BlackChirp::KaiserBessel14,QString("Kaiser-Bessel, B=14"));

    QMenu *winfMenu = m->addMenu(QString("Window Function"));
    QActionGroup *winfGroup = new QActionGroup(winfMenu);
    winfGroup->setExclusive(true);

    for(int i=0; i<winfList.size(); i++)
    {
        QAction *a = winfMenu->addAction(winfList.at(i).second);
        a->setCheckable(true);
        a->setChecked(winfList.at(i).first == d_currentWinf);
        connect(a,&QAction::triggered,this,[=](){ setWinf(winfList.at(i).first); });
    }
    winfMenu->addActions(winfGroup->actions());

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
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString path = s.value(QString("exportPath"),QDir::homePath()).toString();

    int num = d_number;
    if(num < 0)
        num = 0;

    QString name = QFileDialog::getSaveFileName(this,QString("Export Experiment"),path + QString("/ft%1.txt").arg(num));

    if(name.isEmpty())
        return;

    QFile f(name);
    if(!f.open(QIODevice::WriteOnly))
    {
        QMessageBox::critical(this,QString("Export Failed"),QString("Could not open file %1 for writing. Please choose a different filename.").arg(name));
        return;
    }

    QApplication::setOverrideCursor(Qt::BusyCursor);

    f.write(QString("freq%1\tft%1").arg(d_number).toLatin1());

    for(int i=0;i<d_currentFt.size();i++)
        f.write(QString("\n%1\t%2").arg(d_currentFt.at(i).x(),0,'f',6).arg(d_currentFt.at(i).y(),0,'e',12).toLatin1());
    f.close();

    QApplication::restoreOverrideCursor();

    QString newPath = QFileInfo(name).dir().absolutePath();
    s.setValue(QString("exportPath"),newPath);

}

void FtPlot::configureUnits(BlackChirp::FtPlotUnits u)
{
    d_currentUnits = u;
    QwtText title = axisTitle(QwtPlot::yLeft);
    double scf = 1.0;

    switch(u)
    {
    case BlackChirp::FtPlotV:
        title.setText(QString("FT (V)"));
        scf = 1.0;
        break;
    case BlackChirp::FtPlotmV:
        title.setText(QString("FT (mV)"));
        scf = 1e3;
        break;
    case BlackChirp::FtPlotuV:
        title.setText(QString::fromUtf16(u"FT (µV)"));
        scf = 1e6;
        break;
    case BlackChirp::FtPlotnV:
        title.setText(QString("FT (nV)"));
        scf = 1e9;
        break;
    default:
        break;
    }

    QSettings s;
    s.setValue(QString("ftUnits"),d_currentUnits);

    setAxisTitle(QwtPlot::yLeft,title);
    emit unitsChanged(scf);
}

void FtPlot::setWinf(BlackChirp::FtWindowFunction wf)
{
    d_currentWinf = wf;
    QSettings s;
    s.setValue(QString("windowFunction"),d_currentWinf);
    s.sync();

    emit winfChanged(d_currentWinf);
}
