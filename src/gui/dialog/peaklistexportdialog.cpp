#include "peaklistexportdialog.h"
#include "ui_peaklistexportdialog.h"

#include <QSettings>
#include <QFileDialog>
#include <QMessageBox>
#include <QTextStream>

#include <src/data/datastructs.h>

PeakListExportDialog::PeakListExportDialog(const QList<QPointF> peakList, int number, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::PeakListExportDialog), d_number(number), d_peakList(peakList)
{
    ui->setupUi(this);

    connect(ui->ftbRadioButton,&QRadioButton::toggled,ui->ftbOptionsBox,&QGroupBox::setEnabled);
    ui->ftbOptionsBox->setEnabled(false);

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("peakListExport"));
    bool ascii = s.value(QString("ascii"),false).toBool();
    if(ascii)
        ui->asciiRadioButton->setChecked(true);
    else
        ui->ftbRadioButton->setChecked(true);

    bool dipoleEn = s.value(QString("dipoleEnabled"),true).toBool();
    double dipole = s.value(QString("dipole"),1.0).toDouble();
    ui->dipoleDoubleSpinBox->setValue(dipole);
    ui->dipoleCheckBox->setChecked(dipoleEn);
    ui->dipoleDoubleSpinBox->setEnabled(dipole);
    connect(ui->dipoleCheckBox,&QCheckBox::toggled,ui->dipoleDoubleSpinBox,&QDoubleSpinBox::setEnabled);

    bool drOnly = s.value(QString("drOnlyEnabled"),false).toBool();
    double drOnlyThresh = s.value(QString("drOnlyThresh"),1.0).toDouble();
    ui->drOnlyCheckBox->setChecked(drOnly);
    ui->drOnlyThreshSpinBox->setValue(drOnlyThresh);
    ui->drOnlyThreshSpinBox->setEnabled(drOnly);
    connect(ui->drOnlyCheckBox,&QCheckBox::toggled,ui->drOnlyThreshSpinBox,&QDoubleSpinBox::setEnabled);

    int defaultShots = s.value(QString("defaultShots"),100).toInt();
    ui->defaultShotsSpinBox->setValue(defaultShots);

    bool drPowerEnabled = s.value(QString("drPowerEnabled"),false).toBool();
    double drPower = s.value(QString("drPower"),17.0).toDouble();
    ui->drPowerCheckBox->setChecked(drPowerEnabled);
    ui->drPowerDoubleSpinBox->setValue(drPower);
    ui->drPowerDoubleSpinBox->setEnabled(drPowerEnabled);
    connect(ui->drPowerCheckBox,&QCheckBox::toggled,ui->drPowerDoubleSpinBox,&QDoubleSpinBox::setEnabled);

    p_sm = new ShotsModel(this);
    ui->shotsTableView->setModel(p_sm);

    QList<QPair<int,double>> shotsList;
    int num = s.beginReadArray(QString("shotsTable"));
    for(int i=0; i<num; i++)
    {
        s.setArrayIndex(i);
        int shots = s.value(QString("shots"),100).toInt();
        double intensity = s.value(QString("intensity"),1.0).toDouble();
        shotsList.append(qMakePair(shots,intensity));
    }
    s.endArray();
    s.endGroup();
    p_sm->setList(shotsList);

    connect(ui->shotsTableView->selectionModel(),&QItemSelectionModel::selectionChanged,this,&PeakListExportDialog::toggleButtons);
    connect(ui->addShotButton,&QToolButton::clicked,p_sm,&ShotsModel::addEntry);
    connect(ui->insertShotButton,&QToolButton::clicked,this,&PeakListExportDialog::insertShot);
    connect(ui->removeShotButton,&QToolButton::clicked,this,&PeakListExportDialog::removeShots);

    p_pm = new PeakListModel(this);
    p_proxy = new QSortFilterProxyModel(this);
    p_proxy->setSourceModel(p_pm);
    p_proxy->setSortRole(Qt::EditRole);
    p_pm->setPeakList(d_peakList);
    ui->peakListTableView->setModel(p_proxy);
    ui->peakListTableView->sortByColumn(1,Qt::DescendingOrder);
    ui->peakListTableView->setSortingEnabled(true);
    connect(ui->removePeakButton,&QToolButton::clicked,this,&PeakListExportDialog::removePeaks);
    connect(ui->resetPeakListButton,&QPushButton::clicked,[=](){ p_pm->setPeakList(d_peakList);});



}

PeakListExportDialog::~PeakListExportDialog()
{
    delete ui;
}

void PeakListExportDialog::toggleButtons()
{
    QModelIndexList l = ui->shotsTableView->selectionModel()->selectedRows();
    ui->removeShotButton->setEnabled(!l.isEmpty());
    ui->insertShotButton->setEnabled(!l.isEmpty());

    l = ui->peakListTableView->selectionModel()->selectedRows();
    ui->removePeakButton->setEnabled(!l.isEmpty());
}

void PeakListExportDialog::insertShot()
{
    QModelIndexList l = ui->shotsTableView->selectionModel()->selectedRows();
    if(!l.isEmpty())
        p_sm->insertEntry(l.constFirst().row());
}

void PeakListExportDialog::removeShots()
{
    QModelIndexList l = ui->shotsTableView->selectionModel()->selectedRows();
    QList<int> rows;
    for(int i=0; i<l.size(); i++)
        rows.append(l.at(i).row());
    p_sm->removeEntries(rows);
}

void PeakListExportDialog::removePeaks()
{
    QModelIndexList l = ui->peakListTableView->selectionModel()->selectedRows();
    QList<int> rows;
    for(int i=0; i<l.size(); i++)
        rows.append(l.at(i).row());
    p_pm->removePeaks(rows);
}


void PeakListExportDialog::accept()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString savePath = BlackChirp::getExportDir();

    QString ext = QString(".txt");
    if(ui->ftbRadioButton->isChecked())
        ext = QString(".ftb");

    QString name = QFileDialog::getSaveFileName(this,QString("Export Peak List"),
                                                savePath + QString("/peaks%1").arg(d_number) + ext);

    QFile f(name);
    if(!f.open(QIODevice::WriteOnly))
    {
        QMessageBox::critical(this,QString("Export Failed"),QString("Could not open file %1 for writing. Please choose a different filename.").arg(name));
        return;
    }

    QTextStream t(&f);
    QString nl("\n");

    QList<QPair<int,double>> shotsList = p_sm->shotsList();
    std::sort(shotsList.begin(),shotsList.end());

    if(ui->asciiRadioButton->isChecked())
    {
        QString tab("\t");
        t.setRealNumberNotation(QTextStream::ScientificNotation);
        t.setRealNumberPrecision(4);

        for(int i=0; i<p_proxy->rowCount(); i++)
        {
            QModelIndex ind = p_proxy->mapToSource(p_proxy->index(i,0));
            t << QString::number(p_pm->data(p_pm->index(ind.row(),0),Qt::EditRole).toDouble(),'f',4) << tab <<
                 p_pm->data(p_pm->index(ind.row(),1),Qt::EditRole).toDouble() << nl;
        }
    }
    else
    {
        QString dipoleText("");
        QString shotsText("shots:");
        QString drText = QString("drpower:%1").arg(ui->drPowerDoubleSpinBox->value(),0,'f',2);
        QString space(" ");
        t.setRealNumberNotation(QTextStream::FixedNotation);
        t.setRealNumberPrecision(3);

        if(ui->dipoleCheckBox->isChecked())
            dipoleText = QString("dipole:%1").arg(ui->dipoleDoubleSpinBox->value(),0,'f',2);

        t << QString("#File generated by BlackChirp") << nl;

        for(int i=0; i<p_proxy->rowCount(); i++)
        {
            t << nl;
            QModelIndex ind = p_proxy->mapToSource(p_proxy->index(i,0));
            double freq = p_pm->data(p_pm->index(ind.row(),0),Qt::EditRole).toDouble();
            double intensity = p_pm->data(p_pm->index(ind.row(),1),Qt::EditRole).toDouble();

            int shots = ui->defaultShotsSpinBox->value();
            for(int j=0; j<shotsList.size(); j++)
            {
                if(intensity < shotsList.at(j).second)
                    shots = shotsList.at(j).first;
            }


            if(ui->drOnlyCheckBox->isChecked() && intensity<ui->drOnlyThreshSpinBox->value())
            {
                t << QString("amdor drfreq:") << freq << space;

                if(ui->drPowerCheckBox->isChecked())
                    t << drText << space;

                t << QString("#intensity %1").arg(intensity,0,'e',3);
            }
            else
            {
                t << QString("ftmfreq:") << freq << space << shotsText << shots << space;
                if(ui->dipoleCheckBox->isChecked())
                    t << dipoleText << space;
                if(ui->drPowerCheckBox->isChecked())
                    t << drText << space;

                t  << QString("#intensity %1").arg(intensity,0,'e',3);
            }
        }
    }

    t.flush();
    f.close();

    BlackChirp::setExportDir(name);

    s.beginGroup(QString("peakListExport"));
    s.setValue(QString("ascii"),ui->asciiRadioButton->isChecked());
    s.setValue(QString("dipoleEnabled"),ui->dipoleCheckBox->isChecked());
    s.setValue(QString("dipole"),ui->dipoleDoubleSpinBox->value());
    s.setValue(QString("drOnlyEnabled"),ui->drOnlyCheckBox->isChecked());
    s.setValue(QString("drOnlyThresh"),ui->drOnlyThreshSpinBox->value());
    s.setValue(QString("defaultShots"),ui->defaultShotsSpinBox->value());
    s.setValue(QString("drPower"),ui->drPowerDoubleSpinBox->value());
    s.setValue(QString("drPowerEnabled"),ui->drPowerCheckBox->isChecked());

    s.beginWriteArray(QString("shotsTable"));
    for(int i=0; i<shotsList.size(); i++)
    {
        s.setArrayIndex(i);
        s.setValue(QString("shots"),shotsList.at(i).first);
        s.setValue(QString("intensity"),shotsList.at(i).second);
    }
    s.endArray();
    s.endGroup();

    QDialog::accept();
}

ShotsModel::ShotsModel(QObject *parent) : QAbstractTableModel(parent)
{

}

void ShotsModel::setList(const QList<QPair<int, double> > l)
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

QList<QPair<int, double> > ShotsModel::shotsList() const
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

void ShotsModel::removeEntries(QList<int> rows)
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
                return QString("Shots");
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

    return 0;
}
