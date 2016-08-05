#include "peakfindwidget.h"
#include "ui_peakfindwidget.h"

#include <QSettings>
#include <QThread>
#include <QDialog>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QDialogButtonBox>

#include "peaklistexportdialog.h"

PeakFindWidget::PeakFindWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PeakFindWidget), d_number(0), d_busy(false), d_waiting(false)
{
    ui->setupUi(this);

    setEnabled(false);

    p_thread = new QThread(this);
    p_pf = new PeakFinder;
    connect(p_pf,&PeakFinder::peakList,this,&PeakFindWidget::newPeakList);
    connect(p_thread,&QThread::finished,p_pf,&PeakFinder::deleteLater);
    p_pf->moveToThread(p_thread);
    p_thread->start();

    p_listModel = new PeakListModel(this);
    p_proxy = new QSortFilterProxyModel(this);
    p_proxy->setSourceModel(p_listModel);
    p_proxy->setSortRole(Qt::EditRole);
    ui->peakListTableView->setModel(p_proxy);

    connect(ui->findButton,&QPushButton::clicked,this,&PeakFindWidget::findPeaks);
    connect(ui->peakListTableView->selectionModel(),&QItemSelectionModel::selectionChanged,this,&PeakFindWidget::updateRemoveButton);
    connect(ui->liveUpdateBox,&QCheckBox::toggled,[=](bool b){ if(b) findPeaks(); });
    connect(ui->optionsButton,&QPushButton::clicked,this,&PeakFindWidget::launchOptionsDialog);

}

PeakFindWidget::~PeakFindWidget()
{
    delete ui;
    p_thread->quit();
    p_thread->wait();

}

void PeakFindWidget::prepareForExperiment(const Experiment e)
{
    //clear table
    p_listModel->clearPeakList();

    ui->findButton->setEnabled(false);
    ui->removeButton->setEnabled(false);
    ui->exportButton->setEnabled(false);

    d_currentFt.clear();

    d_busy = false;
    d_waiting = false;

    if(e.ftmwConfig().isEnabled())
    {
        setEnabled(true);

        QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
        d_minFreq = qBound(e.ftmwConfig().ftMin(),s.value(QString("peakFind/minFreq"),e.ftmwConfig().ftMin()).toDouble(),e.ftmwConfig().ftMax());
        d_maxFreq = qBound(e.ftmwConfig().ftMin(),s.value(QString("peakFind/maxFreq"),e.ftmwConfig().ftMax()).toDouble(),e.ftmwConfig().ftMax());
        d_snr = s.value(QString("peakFind/snr"),3.0).toDouble();
        d_winSize = s.value(QString("peakFind/windowSize"),11).toInt();
        d_freqRange = qMakePair(e.ftmwConfig().ftMin(),e.ftmwConfig().ftMax());
        d_polyOrder = s.value(QString("peakFind/polyOrder"),6).toInt();
        if(e.ftmwConfig().type() == BlackChirp::FtmwPeakUp)
            d_number = 0;
        else
            d_number = e.number();

        if(d_winSize < d_polyOrder+1)
            d_winSize = d_polyOrder+1;

        if(!(d_winSize % 2))
            d_winSize++;

        if(d_minFreq > d_maxFreq)
            qSwap(d_minFreq,d_maxFreq);

        QMetaObject::invokeMethod(p_pf,"calcCoefs",Qt::BlockingQueuedConnection,Q_ARG(int,d_winSize),Q_ARG(int,d_polyOrder));
    }
    else
        setEnabled(false);
}

void PeakFindWidget::newFt(const QVector<QPointF> ft)
{
    d_currentFt = ft;

    ui->findButton->setEnabled(true);

    if(ui->liveUpdateBox->isChecked())
        findPeaks();
}

void PeakFindWidget::newPeakList(const QList<QPointF> pl)
{
    //send peak list to model
    p_listModel->setPeakList(pl);
    emit peakList(p_listModel->peakList());

    ui->exportButton->setEnabled(!pl.isEmpty());

    if(d_waiting)
        findPeaks();
    else
        d_busy = false;
}

void PeakFindWidget::findPeaks()
{
    if(d_currentFt.isEmpty())
        return;

    if(!d_busy)
    {
        d_busy = true;
        QMetaObject::invokeMethod(p_pf,"findPeaks",Q_ARG(const QVector<QPointF>,d_currentFt),Q_ARG(double,d_minFreq),Q_ARG(double,d_maxFreq),Q_ARG(double,d_snr));
        d_waiting = false;
    }
    else
        d_waiting = true;
}

void PeakFindWidget::removeSelected()
{
    QModelIndexList l = ui->peakListTableView->selectionModel()->selectedRows();
    if(!l.isEmpty())
    {
        QList<int> rows;
        for(int i=0; i<l.size(); i++)
            rows.append(l.at(i).row());

        p_listModel->removePeaks(rows);
    }

    emit peakList(p_listModel->peakList());
}

void PeakFindWidget::updateRemoveButton()
{
    ui->removeButton->setEnabled(!ui->peakListTableView->selectionModel()->selectedRows().isEmpty());
}

void PeakFindWidget::changeScaleFactor(double scf)
{
    p_listModel->scalingChanged(scf);
    emit peakList(p_listModel->peakList());
}

void PeakFindWidget::launchOptionsDialog()
{
    QDialog d(this);
    d.setWindowTitle(QString("Peak Finding Options"));

    QFormLayout *fl = new QFormLayout;

    QDoubleSpinBox *minBox = new QDoubleSpinBox(&d);
    minBox->setDecimals(3);
    minBox->setRange(d_freqRange.first,d_freqRange.second);
    minBox->setValue(d_minFreq);
    minBox->setSuffix(QString(" MHz"));
    fl->addRow(QString("Min Frequency"),minBox);

    QDoubleSpinBox *maxBox = new QDoubleSpinBox(&d);
    maxBox->setDecimals(3);
    maxBox->setRange(d_freqRange.first,d_freqRange.second);
    maxBox->setValue(d_maxFreq);
    maxBox->setSuffix(QString(" MHz"));
    fl->addRow(QString("Max Frequency"),maxBox);

    QDoubleSpinBox *snrBox = new QDoubleSpinBox(&d);
    snrBox->setDecimals(1);
    snrBox->setRange(1.0,10000.0);
    snrBox->setValue(d_snr);
    snrBox->setToolTip(QString("Signal-to-noise ratio threshold for peak detection."));
    fl->addRow(QString("SNR"),snrBox);

    QSpinBox *winBox = new QSpinBox(&d);
    winBox->setRange(7,10001);
    winBox->setSingleStep(2);
    winBox->setValue(d_winSize);
    winBox->setToolTip(QString("Window size for Savitsky-Golay smoothing. Must be odd and greater than the polynomial order."));
    fl->addRow(QString("Window Size"),winBox);

    QSpinBox *orderBox = new QSpinBox(&d);
    orderBox->setRange(2,100);
    orderBox->setValue(d_polyOrder);
    orderBox->setToolTip(QString("Polynomial order for Savistsky-Golay smoothing. Must be less than the window size."));
    fl->addRow(QString("Polynomial Order"),orderBox);

    QVBoxLayout *vbl = new QVBoxLayout;
    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Ok|QDialogButtonBox::Cancel,&d);
    connect(bb->button(QDialogButtonBox::Ok),&QPushButton::clicked,&d,&QDialog::accept);
    connect(bb->button(QDialogButtonBox::Cancel),&QPushButton::clicked,&d,&QDialog::reject);

    vbl->addLayout(fl,1);
    vbl->addWidget(bb,0);

    d.setLayout(vbl);

    if(d.exec() == QDialog::Accepted)
    {
        double minFreq = minBox->value();
        double maxFreq = maxBox->value();
        int ws = winBox->value();
        int po = orderBox->value();

        if(ws < po+1)
            ws = po+1;

        if(!(ws % 2))
            ws++;

        if(minFreq > maxFreq)
            qSwap(d_minFreq,d_maxFreq);

        d_minFreq = minFreq;
        d_maxFreq = maxFreq;
        d_winSize = ws;
        d_polyOrder = po;
        d_snr = snrBox->value();

        QMetaObject::invokeMethod(p_pf,"calcCoefs",Qt::BlockingQueuedConnection,Q_ARG(int,d_winSize),Q_ARG(int,d_polyOrder));

        QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
        s.setValue(QString("peakFind/minFreq"),d_minFreq);
        s.setValue(QString("peakFind/maxFreq"),d_maxFreq);
        s.setValue(QString("peakFind/snr"),d_snr);
        s.setValue(QString("peakFind/windowSize"),d_winSize);
        s.setValue(QString("peakFind/polyOrder"),d_polyOrder);
        s.sync();
    }

}

void PeakFindWidget::launchExportDialog()
{
    PeakListExportDialog d(p_listModel->peakList(),d_number,this);
    d.exec();
}
