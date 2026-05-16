#include "peaklistexportdialog.h"
#include <gui/style/themecolors.h>

#include <QFileDialog>
#include <QMessageBox>
#include <QTextStream>
#include <QSaveFile>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QRadioButton>
#include <QGroupBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QToolButton>
#include <QLabel>
#include <QTableView>
#include <QHeaderView>
#include <QPushButton>
#include <QDialogButtonBox>
#include <QItemSelectionModel>

#include <data/storage/blackchirpcsv.h>

PeakListExportDialog::PeakListExportDialog(const QVector<QPointF> peakList, int number, QWidget *parent) :
    QDialog(parent), SettingsStorage(BC::Key::plExport),
    d_number(number), d_peakList(peakList)
{
    setWindowTitle(QString("Export Peak List"));
    setWindowIcon(ThemeColors::createThemedIcon(":/icons/bc_logo_trans.svg", ThemeColors::IconPrimary, this));

    setupUI();

    // The FTB-options group is only relevant for FTB export; hide it
    // (not just disable) for ASCII so the dialog stays compact, and
    // resize to fit whenever the format changes.
    connect(p_ftbRadio,&QRadioButton::toggled,this,[this](bool ftb){
        p_ftbOptionsBox->setVisible(ftb);
        adjustSize();
    });

    bool ascii = get<bool>(BC::Key::plAscii,true);
    p_asciiRadio->setChecked(ascii);
    p_ftbRadio->setChecked(!ascii);
    p_ftbOptionsBox->setVisible(!ascii);
    registerGetter(BC::Key::plAscii,
                   static_cast<QAbstractButton*>(p_asciiRadio),&QAbstractButton::isChecked);

    bool dipoleEn = get<bool>(BC::Key::plDipoleEn,true);
    double dipole = get<double>(BC::Key::plDipole,1.0);
    p_dipoleBox->setValue(dipole);
    p_dipoleCheck->setChecked(dipoleEn);
    p_dipoleBox->setEnabled(dipoleEn);
    connect(p_dipoleCheck,&QCheckBox::toggled,p_dipoleBox,&QDoubleSpinBox::setEnabled);
    registerGetter(BC::Key::plDipoleEn,
                   static_cast<QAbstractButton*>(p_dipoleCheck),&QAbstractButton::isChecked);
    registerGetter(BC::Key::plDipole,p_dipoleBox,&QDoubleSpinBox::value);

    bool drOnly = get<bool>(BC::Key::plDrOnlyEn,false);
    double drOnlyThresh = get<double>(BC::Key::plDrOnlyThresh,1.0);
    p_drOnlyCheck->setChecked(drOnly);
    p_drOnlyThreshBox->setValue(drOnlyThresh);
    p_drOnlyThreshBox->setEnabled(drOnly);
    connect(p_drOnlyCheck,&QCheckBox::toggled,p_drOnlyThreshBox,&QDoubleSpinBox::setEnabled);
    registerGetter(BC::Key::plDrOnlyEn,
                   static_cast<QAbstractButton*>(p_drOnlyCheck),&QAbstractButton::isChecked);
    registerGetter(BC::Key::plDrOnlyThresh,p_drOnlyThreshBox,&QDoubleSpinBox::value);

    int defaultShots = get<int>(BC::Key::plDefaultShots,100);
    p_defaultShotsBox->setValue(defaultShots);
    registerGetter(BC::Key::plDefaultShots,p_defaultShotsBox,&QSpinBox::value);

    bool drPowerEnabled = get<bool>(BC::Key::plDrPowerEn,false);
    double drPower = get<double>(BC::Key::plDrPower,17.0);
    p_drPowerCheck->setChecked(drPowerEnabled);
    p_drPowerBox->setValue(drPower);
    p_drPowerBox->setEnabled(drPowerEnabled);
    connect(p_drPowerCheck,&QCheckBox::toggled,p_drPowerBox,&QDoubleSpinBox::setEnabled);
    registerGetter(BC::Key::plDrPowerEn,
                   static_cast<QAbstractButton*>(p_drPowerCheck),&QAbstractButton::isChecked);
    registerGetter(BC::Key::plDrPower,p_drPowerBox,&QDoubleSpinBox::value);

    p_sm = new ShotsModel(this);
    p_shotsView->setModel(p_sm);

    QVector<QPair<int,double>> shotsList;
    std::size_t num = getArraySize(BC::Key::plShotsTab);
    for(std::size_t i=0; i<num; ++i)
    {
        int shots = getArrayValue<int>(BC::Key::plShotsTab,i,BC::Key::plShots,100);
        double intensity = getArrayValue<double>(BC::Key::plShotsTab,i,BC::Key::plIntensity,1.0);
        shotsList.append(qMakePair(shots,intensity));
    }
    p_sm->setList(shotsList);

    connect(p_shotsView->selectionModel(),&QItemSelectionModel::selectionChanged,this,&PeakListExportDialog::toggleButtons);
    connect(p_addShotButton,&QToolButton::clicked,p_sm,&ShotsModel::addEntry);
    connect(p_insertShotButton,&QToolButton::clicked,this,&PeakListExportDialog::insertShot);
    connect(p_removeShotButton,&QToolButton::clicked,this,&PeakListExportDialog::removeShots);

    p_pm = new PeakListModel(this);
    p_proxy = new QSortFilterProxyModel(this);
    p_proxy->setSourceModel(p_pm);
    p_proxy->setSortRole(Qt::EditRole);
    p_pm->setPeakList(d_peakList);
    p_peakListView->setModel(p_proxy);
    p_peakListView->sortByColumn(1,Qt::DescendingOrder);
    p_peakListView->setSortingEnabled(true);
    connect(p_removePeakButton,&QToolButton::clicked,this,&PeakListExportDialog::removePeaks);

    adjustSize();
}

void PeakListExportDialog::setupUI()
{
    auto *mainLayout = new QVBoxLayout(this);

    auto *fmtLayout = new QHBoxLayout;
    p_asciiRadio = new QRadioButton("ASCII",this);
    p_ftbRadio = new QRadioButton("FTB",this);
    fmtLayout->addWidget(p_asciiRadio);
    fmtLayout->addWidget(p_ftbRadio);
    mainLayout->addLayout(fmtLayout);

    p_ftbOptionsBox = new QGroupBox("FTB Options",this);
    auto *ftbLayout = new QVBoxLayout(p_ftbOptionsBox);

    auto *shotsRow = new QHBoxLayout;
    shotsRow->addWidget(new QLabel("Default Shots",p_ftbOptionsBox));
    p_defaultShotsBox = new QSpinBox(p_ftbOptionsBox);
    p_defaultShotsBox->setRange(1,100000000);
    p_defaultShotsBox->setSingleStep(20);
    p_defaultShotsBox->setValue(100);
    p_defaultShotsBox->setToolTip("The number of shots to place into the ftb file for strong lines.\n"
                                  "For weak lines, use the table below to set intensity thresholds for more shots.");
    shotsRow->addWidget(p_defaultShotsBox);
    ftbLayout->addLayout(shotsRow);

    auto *dipoleRow = new QHBoxLayout;
    p_dipoleCheck = new QCheckBox("Dipole",p_ftbOptionsBox);
    p_dipoleBox = new QDoubleSpinBox(p_ftbOptionsBox);
    p_dipoleBox->setRange(0.01,10.0);
    p_dipoleBox->setSingleStep(0.5);
    p_dipoleBox->setValue(1.0);
    dipoleRow->addWidget(p_dipoleCheck);
    dipoleRow->addWidget(p_dipoleBox);
    ftbLayout->addLayout(dipoleRow);

    auto *drPowerRow = new QHBoxLayout;
    p_drPowerCheck = new QCheckBox("DR Power",p_ftbOptionsBox);
    p_drPowerBox = new QDoubleSpinBox(p_ftbOptionsBox);
    p_drPowerBox->setSuffix(" dBm");
    p_drPowerBox->setRange(-100.0,100.0);
    drPowerRow->addWidget(p_drPowerCheck);
    drPowerRow->addWidget(p_drPowerBox);
    ftbLayout->addLayout(drPowerRow);

    auto *drOnlyRow = new QHBoxLayout;
    p_drOnlyCheck = new QCheckBox("DR Only Threshold",p_ftbOptionsBox);
    p_drOnlyThreshBox = new QDoubleSpinBox(p_ftbOptionsBox);
    p_drOnlyThreshBox->setDecimals(6);
    p_drOnlyThreshBox->setMaximum(100000.0);
    drOnlyRow->addWidget(p_drOnlyCheck);
    drOnlyRow->addWidget(p_drOnlyThreshBox);
    ftbLayout->addLayout(drOnlyRow);

    auto *siLabel = new QLabel("Shots/Intensity",p_ftbOptionsBox);
    siLabel->setAlignment(Qt::AlignCenter);
    ftbLayout->addWidget(siLabel);

    p_shotsView = new QTableView(p_ftbOptionsBox);
    p_shotsView->setAlternatingRowColors(true);
    p_shotsView->verticalHeader()->setVisible(false);
    p_shotsView->setMinimumHeight(100);
    ftbLayout->addWidget(p_shotsView);

    auto *shotsBtnRow = new QHBoxLayout;
    p_addShotButton = new QToolButton(p_ftbOptionsBox);
    p_addShotButton->setText("Add");
    p_addShotButton->setIcon(ThemeColors::createThemedIcon(":/icons/plus.svg", ThemeColors::IconPrimary, this));
    p_addShotButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    p_addShotButton->setToolTip("Add a shots/intensity row");
    p_insertShotButton = new QToolButton(p_ftbOptionsBox);
    p_insertShotButton->setText("Insert");
    p_insertShotButton->setIcon(ThemeColors::createThemedIcon(":/icons/go-last.svg", ThemeColors::IconSecondary, this));
    p_insertShotButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    p_insertShotButton->setToolTip("Insert a shots/intensity row before the selected row");
    p_removeShotButton = new QToolButton(p_ftbOptionsBox);
    p_removeShotButton->setText("Remove");
    p_removeShotButton->setIcon(ThemeColors::createThemedIcon(":/icons/minus.svg", ThemeColors::IconPrimary, this));
    p_removeShotButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    p_removeShotButton->setToolTip("Remove the selected shots/intensity rows");
    shotsBtnRow->addWidget(p_addShotButton);
    shotsBtnRow->addWidget(p_insertShotButton);
    shotsBtnRow->addWidget(p_removeShotButton);
    ftbLayout->addLayout(shotsBtnRow);

    // FTB options sit beside the peak list (and disappear for ASCII),
    // so toggling the format grows the dialog in width rather than
    // stacking a tall options group above the table.
    auto *contentLayout = new QHBoxLayout;
    contentLayout->addWidget(p_ftbOptionsBox);

    auto *peakColumn = new QVBoxLayout;
    auto *peaksLabel = new QLabel("Peaks",this);
    peaksLabel->setAlignment(Qt::AlignCenter);
    peakColumn->addWidget(peaksLabel);

    p_peakListView = new QTableView(this);
    p_peakListView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    p_peakListView->setAlternatingRowColors(true);
    // Extended (not Multi) selection matches the PeakFindWidget table: a
    // plain click selects a single row while Ctrl/Shift extend it.
    p_peakListView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    p_peakListView->setSelectionBehavior(QAbstractItemView::SelectRows);
    p_peakListView->setMinimumHeight(200);
    peakColumn->addWidget(p_peakListView,1);

    auto *peakBtnRow = new QHBoxLayout;
    auto *resetButton = new QPushButton("Reset",this);
    p_removePeakButton = new QToolButton(this);
    p_removePeakButton->setIcon(ThemeColors::createThemedIcon(":/icons/minus.svg", ThemeColors::IconPrimary, this));
    p_removePeakButton->setToolTip("Remove the selected peaks from the list");
    peakBtnRow->addWidget(resetButton);
    peakBtnRow->addWidget(p_removePeakButton);
    peakBtnRow->addStretch(1);
    peakColumn->addLayout(peakBtnRow);

    contentLayout->addLayout(peakColumn,1);
    mainLayout->addLayout(contentLayout,1);

    connect(resetButton,&QPushButton::clicked,this,[this](){ p_pm->setPeakList(d_peakList);});

    auto *bb = new QDialogButtonBox(QDialogButtonBox::Cancel|QDialogButtonBox::Ok,this);
    connect(bb,&QDialogButtonBox::accepted,this,&PeakListExportDialog::accept);
    connect(bb,&QDialogButtonBox::rejected,this,&PeakListExportDialog::reject);
    mainLayout->addWidget(bb);
}

PeakListExportDialog::~PeakListExportDialog()
{
}

void PeakListExportDialog::toggleButtons()
{
    QModelIndexList l = p_shotsView->selectionModel()->selectedRows();
    p_removeShotButton->setEnabled(!l.isEmpty());
    p_insertShotButton->setEnabled(!l.isEmpty());

    l = p_peakListView->selectionModel()->selectedRows();
    p_removePeakButton->setEnabled(!l.isEmpty());
}

void PeakListExportDialog::insertShot()
{
    QModelIndexList l = p_shotsView->selectionModel()->selectedRows();
    if(!l.isEmpty())
        p_sm->insertEntry(l.constFirst().row());
}

void PeakListExportDialog::removeShots()
{
    QModelIndexList l = p_shotsView->selectionModel()->selectedRows();
    QVector<int> rows;
    for(int i=0; i<l.size(); i++)
        rows.append(l.at(i).row());
    p_sm->removeEntries(rows);
}

void PeakListExportDialog::removePeaks()
{
    QModelIndexList l = p_peakListView->selectionModel()->selectedRows();
    QVector<int> rows;
    for(int i=0; i<l.size(); i++)
        rows.append(l.at(i).row());
    p_pm->removePeaks(rows);
}


void PeakListExportDialog::accept()
{
    // Persist the shots table regardless of export format or cancel so
    // edits to it are never silently discarded.
    {
        QVector<QPair<int,double>> sl = p_sm->shotsList();
        std::sort(sl.begin(),sl.end());
        std::vector<SettingsMap> l;
        l.reserve(sl.size());
        for(int i=0; i<sl.size(); i++)
            l.push_back({ {BC::Key::plShots,sl.at(i).first},
                          {BC::Key::plIntensity,sl.at(i).second} });
        setArray(BC::Key::plShotsTab,l,false);
    }

    QDir d = BlackchirpCSV::textExportDir();

    QString ext = QString(".txt");
    if(p_ftbRadio->isChecked())
        ext = QString(".ftb");

    QString fn = QString("peaks")+ext;
    if(d_number > 0)
        fn = QString("peaks%1").arg(d_number)+ext;
    QString name = QFileDialog::getSaveFileName(this,QString("Export Peak List"),
                                                d.absoluteFilePath(fn));

    // An empty name means the save dialog was cancelled; leave this
    // dialog open without raising a spurious export-failure box.
    if(name.isEmpty())
        return;

    QSaveFile f(name);
    if(!f.open(QIODevice::WriteOnly|QIODevice::Text))
    {
        QMessageBox::critical(this,QString("Export Failed"),QString("Could not open file %1 for writing. Please choose a different filename.").arg(name));
        return;
    }

    QTextStream t(&f);
    QString nl("\n");


    if(p_asciiRadio->isChecked())
    {
        BlackchirpCSV::writeLine(t,{"Frequency","Intensity"});
        for(int i=0; i<p_proxy->rowCount(); i++)
        {
            QModelIndex ind = p_proxy->mapToSource(p_proxy->index(i,0));
            BlackchirpCSV::writeLine(t,{p_pm->data(p_pm->index(ind.row(),0),Qt::EditRole),
                                        p_pm->data(p_pm->index(ind.row(),1),Qt::EditRole)});
        }
    }
    else
    {
        QVector<QPair<int,double>> shotsList = p_sm->shotsList();
        std::sort(shotsList.begin(),shotsList.end());

        QString dipoleText("");
        QString shotsText("shots:");
        QString drText = QString("drpower:%1").arg(p_drPowerBox->value(),0,'f',2);
        QString space(" ");
        t.setRealNumberNotation(QTextStream::FixedNotation);
        t.setRealNumberPrecision(3);

        if(p_dipoleCheck->isChecked())
            dipoleText = QString("dipole:%1").arg(p_dipoleBox->value(),0,'f',2);

        t << QString("#File generated by Blackchirp") << nl;

        for(int i=0; i<p_proxy->rowCount(); i++)
        {
            t << nl;
            QModelIndex ind = p_proxy->mapToSource(p_proxy->index(i,0));
            double freq = p_pm->data(p_pm->index(ind.row(),0),Qt::EditRole).toDouble();
            double intensity = p_pm->data(p_pm->index(ind.row(),1),Qt::EditRole).toDouble();

            int shots = p_defaultShotsBox->value();
            for(int j=0; j<shotsList.size(); j++)
            {
                if(intensity < shotsList.at(j).second)
                    shots = shotsList.at(j).first;
            }


            if(p_drOnlyCheck->isChecked() && intensity<p_drOnlyThreshBox->value())
            {
                t << QString("amdor drfreq:") << freq << space;

                if(p_drPowerCheck->isChecked())
                    t << drText << space;

                t << QString("#intensity %1").arg(intensity,0,'e',3);
            }
            else
            {
                t << QString("ftmfreq:") << freq << space << shotsText << shots << space;
                if(p_dipoleCheck->isChecked())
                    t << dipoleText << space;
                if(p_drPowerCheck->isChecked())
                    t << drText << space;

                t  << QString("#intensity %1").arg(intensity,0,'e',3);
            }
        }
    }

    f.commit();
    QDialog::accept();
}

ShotsModel::ShotsModel(QObject *parent) : QAbstractTableModel(parent)
{

}

void ShotsModel::setList(const QVector<QPair<int, double> > l)
{
    if(!d_shotsList.isEmpty())
    {
        beginRemoveRows(QModelIndex(),0,d_shotsList.size()-1);
        d_shotsList.clear();
        endRemoveRows();
    }

    if(!l.isEmpty())
    {
        beginInsertRows(QModelIndex(),0,l.size()-1);
        d_shotsList = l;
        endInsertRows();
    }
}

QVector<QPair<int, double> > ShotsModel::shotsList() const
{
    return d_shotsList;
}

void ShotsModel::addEntry()
{
    insertEntry(d_shotsList.size());
}

void ShotsModel::insertEntry(int pos)
{
    if(pos < 0 || pos > d_shotsList.size())
        pos = d_shotsList.size();

    beginInsertRows(QModelIndex(),pos,pos);
    d_shotsList.insert(pos,qMakePair(100,1.0));
    endInsertRows();
}

void ShotsModel::removeEntries(QVector<int> rows)
{
    std::sort(rows.begin(),rows.end());
    for(int i=rows.size()-1; i>=0; i--)
    {
        if(rows.at(i) < 0 || rows.at(i) >= d_shotsList.size())
            continue;

        beginRemoveRows(QModelIndex(),rows.at(i),rows.at(i));
        d_shotsList.removeAt(rows.at(i));
        endRemoveRows();
    }
}


int ShotsModel::rowCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return d_shotsList.size();
}

int ShotsModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return 2;
}

QVariant ShotsModel::data(const QModelIndex &index, int role) const
{
    if(index.row() < 0 || index.row() >= d_shotsList.size())
        return QVariant();

    if(role == Qt::DisplayRole)
    {
        if(index.column() == 0)
            return QString::number(d_shotsList.at(index.row()).first);
        else if(index.column() == 1)
            return QString::number(d_shotsList.at(index.row()).second,'e',3);
    }
    else if(role == Qt::EditRole)
    {
        if(index.column() == 0)
            return d_shotsList.at(index.row()).first;
        else if(index.column() == 1)
            return d_shotsList.at(index.row()).second;
    }

    return QVariant();
}

bool ShotsModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if(index.row() < 0 || index.row() >= d_shotsList.size())
        return false;

    if(role == Qt::EditRole)
    {
        bool ok = false;
        if(index.column() == 0)
        {
            int i = value.toInt(&ok);
            if(ok)
                d_shotsList[index.row()].first = i;
        }
        else if(index.column() == 1)
        {
            double d = value.toDouble(&ok);
            if(ok)
                d_shotsList[index.row()].second = d;
        }

        if(ok)
            emit dataChanged(index,index);

        return ok;
    }


    return false;
}

QVariant ShotsModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if(orientation == Qt::Horizontal)
    {
        if(role == Qt::DisplayRole)
        {
            if(section == 0)
                return BC::Key::plShots;
            else if(section == 1)
                return QString("Intensity Limit");
        }
        else if(role == Qt::ToolTipRole)
        {
            if(section == 0)
                return QString("The number of shots to use for a peak in this intensity range.");
            else if(section == 1)
                return QString("Upper limit on intensity for this number of shots.\nIf the intensity of a line is greater than any entry in this table, the value in the \"Default Shots\" box will be used instead.\n\nFor example, if the default shots box is set to 10, and you add entries of 50/1 and 100/0.1,\nthen all lines with intensity >=1 will be set to 10 shots, 0.1 <= intensity < 1 will be set to 50, and < 0.1 will be set to 100.");
        }
    }

    return QVariant();
}

Qt::ItemFlags ShotsModel::flags(const QModelIndex &index) const
{
    if(index.row() >= 0 && index.row() < d_shotsList.size() && index.column() < 2)
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsEditable;

    return Qt::NoItemFlags;
}
