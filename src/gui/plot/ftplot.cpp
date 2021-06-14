#include <src/gui/plot/ftplot.h>

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
#include <QDialogButtonBox>
#include <QPushButton>


#include <qwt6/qwt_scale_widget.h>
#include <qwt6/qwt_scale_div.h>
#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_grid.h>
#include <qwt6/qwt_symbol.h>

FtPlot::FtPlot(const QString id, QWidget *parent) :
    ZoomPanPlot(BC::Key::ftPlot+id,parent), d_number(0), d_id(id), d_currentUnits(BlackChirp::FtPlotV)
{
    //make axis label font smaller
    this->setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    this->setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));

    //build axis titles with small font
    QwtText blabel(QString("Frequency (MHz)"));
    blabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::xBottom,blabel);

    QwtText llabel(QString("FT "+id));
    llabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::yLeft,llabel);

    configureUnits(BlackChirp::FtPlotuV);

    //build and configure curve object
    p_curve = new QwtPlotCurve(QString("FT"));
    setCurveColor(p_curve,BC::Key::ftColor, get<QColor>(BC::Key::ftColor,palette().color(QPalette::BrightText)));
    p_curve->setRenderHint(QwtPlotItem::RenderAntialiased);
    p_curve->attach(this);
    p_curve->setVisible(false);

    p_peakData = new QwtPlotCurve(QString("Peaks"));
    p_peakData->setStyle(QwtPlotCurve::NoCurve);
    p_peakData->setRenderHint(QwtPlotCurve::RenderAntialiased);

    auto c = get<QColor>(BC::Key::peakColor,QColor(Qt::red));
    QwtSymbol *sym = new QwtSymbol(QwtSymbol::Ellipse);
    sym->setSize(5);
    sym->setColor(c);
    sym->setPen(QPen(c));
    p_peakData->setSymbol(sym);

    p_peakData->attach(this);

    p_plotGrid = new QwtPlotGrid();
    p_plotGrid->enableX(true);
    p_plotGrid->enableXMin(true);
    p_plotGrid->enableY(true);
    p_plotGrid->enableYMin(true);
    QPen p;
    p.setColor(get<QColor>(BC::Key::gridColor,palette().color(QPalette::Light)));
    p.setStyle(Qt::DashLine);
    p_plotGrid->setMajorPen(p);
    p.setStyle(Qt::DotLine);
    p_plotGrid->setMinorPen(p);
    p_plotGrid->attach(this);

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

    d_currentFt = Ft();
    p_curve->setSamples(QVector<QPointF>());
    p_peakData->setSamples(QVector<QPointF>());

    setAxisAutoScaleRange(QwtPlot::yLeft,0.0,1.0);
    if(!c.isEnabled())
    {
        p_curve->setVisible(false);
        setAxisAutoScaleRange(QwtPlot::xBottom,0.0,1.0);
    }
    else
    {
        p_curve->setVisible(true);
        setAxisAutoScaleRange(QwtPlot::xBottom,c.ftMinMHz(),c.ftMaxMHz());
    }
    autoScale();
    QSettings s;
    configureUnits(static_cast<BlackChirp::FtPlotUnits>(s.value(QString("ftUnits"),BlackChirp::FtPlotmV).toInt()));
}

Ft FtPlot::currentFt() const
{
    return d_currentFt;
}

void FtPlot::newFt(const Ft ft)
{
//    if(ft.isEmpty())
//        return;

    d_currentFt = ft;

    if(ft.isEmpty())
    {
        setAxisAutoScaleRange(QwtPlot::yLeft,0.0,0.0);
//        setAxisAutoScaleRange(QwtPlot::xBottom,ft.minFreq(),ft.maxFreq());
    }
    else
    {
        setAxisAutoScaleRange(QwtPlot::yLeft,ft.yMin(),ft.yMax());
        setAxisAutoScaleRange(QwtPlot::xBottom,ft.minFreq(),ft.maxFreq());
    }
    filterData();
    replot();
}

void FtPlot::filterData()
{
    if(d_currentFt.size() < 2)
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
        int numPnts = 0;
        double nextPixelX = map.invTransform(pixel+1.0);
        while(dataIndex+1 < d_currentFt.size() && d_currentFt.at(dataIndex).x() < nextPixelX)
        {
            auto pt = d_currentFt.at(dataIndex);
            min = qMin(pt.y(),min);
            max = qMax(pt.y(),max);

            dataIndex++;
            numPnts++;
        }

        if(numPnts == 1)
            filtered.append(d_currentFt.at(dataIndex-1));
        else if (numPnts > 1)
        {
            QPointF first(map.invTransform(pixel),min);
            QPointF second(map.invTransform(pixel),max);
            filtered.append(first);
            filtered.append(second);
        }
    }

    if(dataIndex < d_currentFt.size())
        filtered.append(d_currentFt.at(dataIndex));

    //assign data to curve object
    p_curve->setSamples(filtered);
}

void FtPlot::buildContextMenu(QMouseEvent *me)
{
    if(d_currentFt.size() < 2 || !isEnabled())
        return;

    QMenu *m = contextMenu();

    QAction *ftColorAction = m->addAction(QString("Change FT Color..."));
    connect(ftColorAction,&QAction::triggered,this,[=](){ setCurveColor(p_curve,BC::Key::ftColor); });

    QAction *gridColorAction = m->addAction(QString("Change Grid Color..."));
    connect(gridColorAction,&QAction::triggered,this,[=](){ changeGridColor(getColor(p_plotGrid->majorPen().color())); });

    QAction *peakColorAction = m->addAction(QString("Change Peak Color..."));
    connect(peakColorAction,&QAction::triggered,this,[=]() { setCurveColor(p_peakData,BC::Key::peakColor); });

    QAction *exportAction = m->addAction(QString("Export XY..."));
    connect(exportAction,&QAction::triggered,this,&FtPlot::exportXY);

    m->popup(me->globalPos());
}

void FtPlot::changeGridColor(QColor c)
{
    if(!c.isValid())
        return;

    set(BC::Key::gridColor,c);

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

    QDialog d(this);
    d.setWindowTitle(QString("Export FT"));
    d.setWindowIcon(QIcon(QString(":/icons/bc_logo_small.png")));

    QVBoxLayout *vbl = new QVBoxLayout;
    QFormLayout *fl = new QFormLayout;

    double min = axisScaleDiv(QwtPlot::xBottom).lowerBound();
    double max = axisScaleDiv(QwtPlot::xBottom).upperBound();

    QDoubleSpinBox *minBox = new QDoubleSpinBox;
    minBox->setRange(d_currentFt.minFreq(),d_currentFt.maxFreq());
    minBox->setDecimals(3);
    minBox->setValue(min);
    minBox->setSuffix(QString(" MHz"));
    minBox->setSingleStep(500.0);
    fl->addRow(QString("Minimum Frequency"),minBox);

    QDoubleSpinBox *maxBox = new QDoubleSpinBox;
    maxBox->setRange(d_currentFt.minFreq(),d_currentFt.maxFreq());
    maxBox->setDecimals(3);
    maxBox->setValue(max);
    maxBox->setSuffix(QString(" MHz"));
    maxBox->setSingleStep(500.0);
    fl->addRow(QString("Maximum Frequency"),maxBox);

    vbl->addLayout(fl,1);

    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(bb->button(QDialogButtonBox::Ok),&QPushButton::clicked,&d,&QDialog::accept);
    connect(bb->button(QDialogButtonBox::Cancel),&QPushButton::clicked,&d,&QDialog::reject);
    vbl->addWidget(bb,0);

    d.setLayout(vbl);

    if(d.exec() == QDialog::Rejected)
        return;

    QString path = BlackChirp::getExportDir();

    int num = d_number;
    if(num < 0)
        num = 0;

    QString name = QFileDialog::getSaveFileName(this,QString("Export FT"),path + QString("/ft%1.txt").arg(num));

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
    {
        if(d_currentFt.at(i).x() >= minBox->value() && d_currentFt.at(i).x() <= maxBox->value())
            f.write(QString("\n%1\t%2").arg(d_currentFt.at(i).x(),0,'f',6)
                    .arg(d_currentFt.at(i).y(),0,'e',12).toLatin1());
    }

    f.close();

    QApplication::restoreOverrideCursor();

    BlackChirp::setExportDir(name);

}

void FtPlot::configureUnits(BlackChirp::FtPlotUnits u)
{
    if(u == d_currentUnits)
        return;

    d_currentUnits = u;
    QwtText title = axisTitle(QwtPlot::yLeft);

    switch(u)
    {
    case BlackChirp::FtPlotV:
        title.setText(QString("FT "+d_id+" (V)"));
        break;
    case BlackChirp::FtPlotmV:
        title.setText(QString("FT "+d_id+" (mV)"));
        break;
    case BlackChirp::FtPlotuV:
        title.setText(QString("FT "+d_id+QString::fromUtf16(u" (µV)")));
        break;
    case BlackChirp::FtPlotnV:
        title.setText(QString("FT "+d_id+" (nV)"));
        break;
    default:
        break;
    }



    setAxisTitle(QwtPlot::yLeft,title);
//    emit unitsChanged(scf);
//    emit scalingChange(scf/oldScf);
}

void FtPlot::newPeakList(const QList<QPointF> l)
{

    p_peakData->setSamples(l.toVector());
    replot();
}
