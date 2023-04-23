#include "peakfindwidget.h"
#include "ui_peakfindwidget.h"

#include <QThread>
#include <QDialog>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QDialogButtonBox>
#include <QtConcurrent/QtConcurrent>

#include <gui/dialog/peaklistexportdialog.h>

PeakFindWidget::PeakFindWidget(Ft ft, int number, QWidget *parent):
    QWidget(parent), SettingsStorage(BC::Key::peakFind),
    ui(new Ui::PeakFindWidget), d_number(number), d_busy(false), d_waiting(false)
{
    ui->setupUi(this);

    p_pf = new PeakFinder(this);
    connect(p_pf,&PeakFinder::peakList,this,&PeakFindWidget::newPeakList);

    p_listModel = new PeakListModel(this);
    p_proxy = new QSortFilterProxyModel(this);
    p_proxy->setSourceModel(p_listModel);
    p_proxy->setSortRole(Qt::EditRole);
    ui->peakListTableView->setModel(p_proxy);

    connect(ui->findButton,&QPushButton::clicked,this,&PeakFindWidget::findPeaks);
    connect(ui->peakListTableView->selectionModel(),&QItemSelectionModel::selectionChanged,this,&PeakFindWidget::updateRemoveButton);
    connect(ui->liveUpdateBox,&QCheckBox::toggled,[=](bool b){ if(b) findPeaks(); });
    connect(ui->optionsButton,&QPushButton::clicked,this,&PeakFindWidget::launchOptionsDialog);
    connect(ui->exportButton,&QPushButton::clicked,this,&PeakFindWidget::launchExportDialog);

    d_minFreq = get<double>(BC::Key::pfMinFreq,ft.minFreqMHz());
    d_maxFreq = get<double>(BC::Key::pfMaxFreq,ft.maxFreqMHz());
    d_snr = get<double>(BC::Key::pfSnr,5.0);
    d_winSize = get<int>(BC::Key::pfWinSize,11);
    d_polyOrder = get<int>(BC::Key::pfOrder,6);

    if(d_minFreq > ft.maxFreqMHz())
        d_minFreq = ft.minFreqMHz();
    if(d_maxFreq < d_minFreq)
        d_maxFreq = ft.maxFreqMHz();

    d_currentFt = ft;

}

PeakFindWidget::~PeakFindWidget()
{
    delete ui;

}

void PeakFindWidget::newFt(const Ft ft)
{
    d_currentFt = ft;

    ui->findButton->setEnabled(true);

    if(ui->liveUpdateBox->isChecked())
        findPeaks();
}

void PeakFindWidget::newPeakList(const QVector<QPointF> pl)
{
    d_busy = false;

    //send peak list to model
    p_listModel->setPeakList(pl);
    ui->peakListTableView->resizeColumnsToContents();
    emit peakList(p_listModel->peakList());

    ui->exportButton->setEnabled(!pl.isEmpty());

    if(d_waiting)
        findPeaks();
}

void PeakFindWidget::findPeaks()
{
    if(d_currentFt.isEmpty())
        return;

    if(!d_busy)
    {
        d_busy = true;
        QtConcurrent::run([this](){p_pf->findPeaks(d_currentFt,d_minFreq,d_maxFreq,d_snr);});
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
        QVector<int> rows;
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
    if(p_listModel->rowCount(QModelIndex()) > 0)
    {
        p_listModel->scalingChanged(scf);
        emit peakList(p_listModel->peakList());
    }
}

void PeakFindWidget::launchOptionsDialog()
{
    QDialog d(this);
    d.setWindowTitle(QString("Peak Finding Options"));

    QFormLayout *fl = new QFormLayout;

    QDoubleSpinBox *minBox = new QDoubleSpinBox(&d);
    minBox->setDecimals(3);
    minBox->setRange(d_currentFt.minFreqMHz(),d_currentFt.maxFreqMHz());
    minBox->setValue(d_minFreq);
    minBox->setSuffix(QString(" MHz"));
    fl->addRow(QString("Min Frequency"),minBox);

    QDoubleSpinBox *maxBox = new QDoubleSpinBox(&d);
    maxBox->setDecimals(3);
    maxBox->setRange(d_currentFt.minFreqMHz(),d_currentFt.maxFreqMHz());
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
            qSwap(minFreq,maxFreq);

        d_minFreq = minFreq;
        d_maxFreq = maxFreq;
        d_winSize = ws;
        d_polyOrder = po;
        d_snr = snrBox->value();

        QMetaObject::invokeMethod(p_pf,[this](){p_pf->calcCoefs(d_winSize,d_polyOrder);});

        set(BC::Key::pfMinFreq,d_minFreq,false);
        set(BC::Key::pfMaxFreq,d_maxFreq,false);
        set(BC::Key::pfSnr,d_snr,false);
        set(BC::Key::pfWinSize,d_winSize,false);
        set(BC::Key::pfOrder,d_polyOrder,false);
        save();
    }

}

void PeakFindWidget::launchExportDialog()
{
    PeakListExportDialog d(p_listModel->peakList(),d_number,this);
    d.exec();
}
