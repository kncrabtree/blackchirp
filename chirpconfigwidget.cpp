#include "chirpconfigwidget.h"
#include "ui_chirpconfigwidget.h"
#include <QSettings>
#include <QMessageBox>

ChirpConfigWidget::ChirpConfigWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ChirpConfigWidget), p_ctm(new ChirpTableModel(this))
{
    ui->setupUi(this);
    ui->chirpTable->setModel(p_ctm);

    connect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::setButtonStates);
    connect(ui->chirpTable->selectionModel(),&QItemSelectionModel::selectionChanged,this,&ChirpConfigWidget::setButtonStates);
    connect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::checkChirp);
    connect(ui->chirpTable->selectionModel(),&QItemSelectionModel::selectionChanged,this,&ChirpConfigWidget::checkChirp);

    initializeFromSettings();
    setButtonStates();

    connect(ui->addButton,&QPushButton::clicked,this,&ChirpConfigWidget::addSegment);
    connect(ui->insertButton,&QPushButton::clicked,this,&ChirpConfigWidget::insertSegment);
    connect(ui->moveUpButton,&QPushButton::clicked,[=](){ moveSegments(-1); });
    connect(ui->moveDownButton,&QPushButton::clicked,[=](){ moveSegments(1); });
    connect(ui->removeButton,&QPushButton::clicked,this,&ChirpConfigWidget::removeSegments);
    connect(ui->clearButton,&QPushButton::clicked,this,&ChirpConfigWidget::clear);

    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    connect(ui->preChirpDelaySpinBox,vc,this,&ChirpConfigWidget::checkChirp);
    connect(ui->protectionDelaySpinBox,vc,this,&ChirpConfigWidget::checkChirp);
    connect(ui->chirpsSpinBox,vc,this,&ChirpConfigWidget::checkChirp);
    connect(ui->chirpsSpinBox,vc,[=](int n){ui->chirpIntervalDoubleSpinBox->setEnabled(n>1);});
    connect(ui->chirpIntervalDoubleSpinBox,static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),this,&ChirpConfigWidget::checkChirp);

    ui->chirpTable->setItemDelegate(new DoubleSpinBoxDelegate);
}

ChirpConfigWidget::~ChirpConfigWidget()
{
    delete ui;
}

void ChirpConfigWidget::initializeFromSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("valonSynth"));
    d_valonFreq = s.value(QString("txFreq"),5760.0).toDouble();
    s.endGroup();

    s.beginGroup(QString("awg"));
    double awgMin = s.value(QString("minFreq"),100.0).toDouble();
    double awgMax = s.value(QString("maxFreq"),6250.0).toDouble();
    s.endGroup();

    s.beginGroup(QString("chirpConfig"));
    d_awgMult = s.value(QString("awgMult"),1.0).toDouble();
    d_valonMult = s.value(QString("valonMult"),2.0).toDouble();
    d_txMult = s.value(QString("txMult"),4.0).toDouble();
    d_txSidebandSign = s.value(QString("txSidebandSign"),-1.0).toDouble();

    d_chirpMinMax.first = qMin(d_txMult*(d_txSidebandSign*d_awgMult*awgMin + d_valonMult*d_valonFreq),d_txMult*(d_txSidebandSign*d_awgMult*awgMax + d_valonMult*d_valonFreq));
    d_chirpMinMax.second = qMax(d_txMult*(d_txSidebandSign*d_awgMult*awgMin + d_valonMult*d_valonFreq),d_txMult*(d_txSidebandSign*d_awgMult*awgMax + d_valonMult*d_valonFreq));
    s.setValue(QString("chirpMin"),d_chirpMinMax.first);
    s.setValue(QString("chirpMax"),d_chirpMinMax.second);
    s.sync();

    ui->preChirpDelaySpinBox->setValue(s.value(QString("preChirpDelay"),300).toInt());
    ui->protectionDelaySpinBox->setValue(s.value(QString("protectionDelay"),300).toInt());
    ui->chirpsSpinBox->setValue(s.value(QString("numChirps"),10).toInt());
    ui->chirpIntervalDoubleSpinBox->setValue(s.value(QString("chirpInterval"),20.0).toDouble());

    clearList();

    int numSegments = s.beginReadArray(QString("segments"));
    for(int i=0; i<numSegments; i++)
    {
        s.setArrayIndex(i);
        double start = qBound(d_chirpMinMax.first,s.value(QString("startFreq"),d_chirpMinMax.first).toDouble(),d_chirpMinMax.second);
        double end = qBound(d_chirpMinMax.first,s.value(QString("endFreq"),d_chirpMinMax.second).toDouble(),d_chirpMinMax.second);
        double dur = qBound(0.1,s.value(QString("duration"),500.0).toDouble(),100000.0);
        p_ctm->addSegment(start,end,dur,p_ctm->rowCount(QModelIndex()));
    }
    s.endArray();
    s.endGroup();

}

void ChirpConfigWidget::enableEditing(bool enabled)
{
    ui->chirpConfigurationBox->setEnabled(enabled);
}

void ChirpConfigWidget::setButtonStates()
{
    //set which buttons should be enabled
    //get selection
    QModelIndexList l = ui->chirpTable->selectionModel()->selectedRows();
    bool c = isSelectionContiguous(l);

    //insert button only enabled if one item is selected
    ui->insertButton->setEnabled(l.size() == 1);

    //remove button active if one or more rows selected
    ui->removeButton->setEnabled(l.size() > 0);

    //move buttons enabled only if selection is contiguous
    ui->moveDownButton->setEnabled(c);
    ui->moveUpButton->setEnabled(c);

    //clear button only enabled if table is not empty
    ui->clearButton->setEnabled(p_ctm->rowCount(QModelIndex()) > 0);
}

void ChirpConfigWidget::addSegment()
{
    p_ctm->addSegment(d_chirpMinMax.first,d_chirpMinMax.second,0.500,-1);
    checkChirp();
}

void ChirpConfigWidget::insertSegment()
{
    QModelIndexList l = ui->chirpTable->selectionModel()->selectedRows();
    if(l.size() != 1)
        return;

    p_ctm->addSegment(d_chirpMinMax.first,d_chirpMinMax.second,0.500,l.at(0).row());
    checkChirp();
}

void ChirpConfigWidget::moveSegments(int direction)
{
    QModelIndexList l = ui->chirpTable->selectionModel()->selectedRows();

    if(l.isEmpty())
        return;

    //get the selected rows
    QList<int> sortList;
    for(int i=0; i<l.size(); i++)
        sortList.append(l.at(i).row());

    qSort(sortList);

    //make sure selection is contiguous
    if(sortList.size()>1 && sortList.at(sortList.size()-1) - sortList.at(0) != sortList.size()-1)
        return;

    p_ctm->moveSegments(sortList.at(0),sortList.at(sortList.size()-1),direction);
    checkChirp();
}

void ChirpConfigWidget::removeSegments()
{
    QModelIndexList l = ui->chirpTable->selectionModel()->selectedRows();

    if(l.isEmpty())
        return;

    QList<int> rows;
    for(int i=0;i<l.size();i++)
        rows.append(l.at(i).row());

    p_ctm->removeSegments(rows);
    checkChirp();
}

void ChirpConfigWidget::clear()
{
    int ret = QMessageBox::question(this,QString("Clear all rows"),
                          QString("Are you sure you want to remove all segments from the table?"),
                          QMessageBox::Yes|QMessageBox::Cancel, QMessageBox::Cancel);

    if(ret == QMessageBox::Yes)
        clearList();
}

void ChirpConfigWidget::checkChirp()
{
    ChirpConfig cc;
    cc.setPreChirpDelay(ui->preChirpDelaySpinBox->value()/1e3);
    cc.setProtectionDelay(ui->protectionDelaySpinBox->value()/1e3);
    cc.setNumChirps(ui->chirpsSpinBox->value());
    cc.setChirpInterval(ui->chirpIntervalDoubleSpinBox->value());

    QList<ChirpConfig::ChirpSegment> l = p_ctm->segmentList();
    for(int i=0; i<l.size();i++)
    {
        l[i].startFreqMHz = d_txSidebandSign*(l.at(i).startFreqMHz/d_txMult - d_valonMult*d_valonFreq)/d_awgMult;
        l[i].endFreqMHz = d_txSidebandSign*(l.at(i).endFreqMHz/d_txMult - d_valonMult*d_valonFreq)/d_awgMult;
        l[i].alphaUs = (l.at(i).endFreqMHz - l.at(i).startFreqMHz)/l.at(i).durationUs;
    }
    cc.setSegmentList(l);



    cc.validate();
    ui->chirpPlot->newChirp(cc);
    emit chirpConfigChanged(cc);
}

bool ChirpConfigWidget::isSelectionContiguous(QModelIndexList l)
{
    if(l.isEmpty())
        return false;

    if(l.size()==1)
        return true;

    //selection is contiguous if the last row minus first row is equal to the size, after the list has been sorted
    QList<int> sortList;
    for(int i=0; i<l.size(); i++)
        sortList.append(l.at(i).row());

    qSort(sortList);

    if(sortList.at(sortList.size()-1) - sortList.at(0) != sortList.size()-1)
        return false;

    return true;
}

void ChirpConfigWidget::clearList()
{
    if(p_ctm->rowCount(QModelIndex()) > 0)
        p_ctm->removeRows(0,p_ctm->rowCount(QModelIndex()),QModelIndex());

    checkChirp();
}
