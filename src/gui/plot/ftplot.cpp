#include <gui/plot/ftplot.h>

#include <QFont>
#include <QMouseEvent>
#include <QEvent>
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
#include <qwt6/qwt_plot_grid.h>


#include <gui/plot/blackchirpplotcurve.h>

FtPlot::FtPlot(const QString id, QWidget *parent) :
    ZoomPanPlot(BC::Key::ftPlot+id,parent), d_number(0), d_id(id)
{

    setPlotAxisTitle(QwtPlot::xBottom,QString("Frequency (MHz)"));
    setPlotAxisTitle(QwtPlot::yLeft,QString("FT "+id));

    //build and configure curve object
    p_curve = new BlackchirpPlotCurve(BC::Key::ftCurve+id);
    p_curve->attach(this);

    p_peakData = new BlackchirpPlotCurve(BC::Key::peakCurve+id,Qt::NoPen,QwtSymbol::Ellipse);
    p_peakData->attach(this);
}

FtPlot::~FtPlot()
{
}

void FtPlot::prepareForExperiment(const Experiment e)
{
    d_number = e.d_number;

    d_currentFt = Ft();
    p_curve->setCurveData(QVector<QPointF>());
    p_peakData->setCurveData(QVector<QPointF>());

    p_curve->setVisible(e.ftmwConfig().isEnabled());

    autoScale();
}

Ft FtPlot::currentFt() const
{
    return d_currentFt;
}

void FtPlot::newFt(const Ft ft)
{
    d_currentFt = ft;
    p_curve->setCurveData(ft.toVector(),ft.yMin(),ft.yMax());
    replot();
}

void FtPlot::buildContextMenu(QMouseEvent *me)
{

    QMenu *m = contextMenu();

    //this will go away soon too
    QAction *exportAction = m->addAction(QString("Export XY..."));
    connect(exportAction,&QAction::triggered,this,&FtPlot::exportXY);

    m->popup(me->globalPos());
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

void FtPlot::configureUnits(FtWorker::FtUnits u)
{

    QwtText title = axisTitle(QwtPlot::yLeft);

    switch(u)
    {
    case FtWorker::FtV:
        title.setText(QString("FT "+d_id+" (V)"));
        break;
    case FtWorker::FtmV:
        title.setText(QString("FT "+d_id+" (mV)"));
        break;
    case FtWorker::FtuV:
        title.setText(QString("FT "+d_id+QString::fromUtf16(u" (µV)")));
        break;
    case FtWorker::FtnV:
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

    if(!l.isEmpty())
        p_peakData->setCurveData(l.toVector());

    p_peakData->setCurveVisible(!l.isEmpty());
    replot();
}
