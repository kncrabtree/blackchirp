#include "chirpconfigwidget.h"
#include "ui_chirpconfigwidget.h"

#include <QMessageBox>
#include <QSpinBox>
#include <QInputDialog>

#include <hardware/core/ftmwdigitizer/ftmwscope.h>

ChirpConfigWidget::ChirpConfigWidget(QWidget *parent) :
    QWidget(parent), SettingsStorage(BC::Key::ChirpConfigWidget),
    ui(new Ui::ChirpConfigWidget), p_ctm(new ChirpTableModel(this)), d_rampOnly(false)
{
    ui->setupUi(this);
    ui->chirpTable->setModel(p_ctm);
    ui->chirpTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    SettingsStorage s(BC::Key::FtmwScope::ftmwScope,SettingsStorage::Hardware);
    bool ff = s.get(BC::Key::Digi::multiRecord,false);

    if(!ff)
    {
        ui->chirpsSpinBox->setValue(1);
        ui->chirpsSpinBox->setEnabled(false);
    }

    QString us = QString::fromUtf16(u" μs");
    double minPreProt = getOrSetDefault(BC::Key::minPreProt,0.0);
    double minPreGate = getOrSetDefault(BC::Key::minPreGate,0.0);
    double minPostGate = getOrSetDefault(BC::Key::minPostGate,-0.5);
    double minPostProt = getOrSetDefault(BC::Key::minPostProt,0.0);

    ui->preChirpProtectionDoubleSpinBox->setMinimum(minPreProt);
    ui->preChirpDelayDoubleSpinBox->setMinimum(minPreGate);
    ui->postChirpDelayDoubleSpinBox->setMinimum(minPostGate);
    ui->postChirpProtectionDoubleSpinBox->setMinimum(minPostProt);

    ui->preChirpDelayDoubleSpinBox->setSuffix(us);
    ui->preChirpProtectionDoubleSpinBox->setSuffix(us);
    ui->postChirpDelayDoubleSpinBox->setSuffix(us);
    ui->postChirpProtectionDoubleSpinBox->setSuffix(us);
    ui->chirpIntervalDoubleSpinBox->setSuffix(us);


    connect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::setButtonStates);
    connect(ui->chirpTable->selectionModel(),&QItemSelectionModel::selectionChanged,this,&ChirpConfigWidget::setButtonStates);
    connect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::updateChirpPlot);
//    connect(ui->chirpTable->selectionModel(),&QItemSelectionModel::selectionChanged,this,&ChirpConfigWidget::checkChirp);

    setButtonStates();



    connect(ui->addButton,&QPushButton::clicked,this,&ChirpConfigWidget::addSegment);
    connect(ui->addEmptyButton,&QPushButton::clicked,this,&ChirpConfigWidget::addEmptySegment);
    connect(ui->insertButton,&QPushButton::clicked,this,&ChirpConfigWidget::insertSegment);
    connect(ui->insertEmptyButton,&QPushButton::clicked,this,&ChirpConfigWidget::insertEmptySegment);
    connect(ui->moveUpButton,&QPushButton::clicked,[=](){ moveSegments(-1); });
    connect(ui->moveDownButton,&QPushButton::clicked,[=](){ moveSegments(1); });
    connect(ui->removeButton,&QPushButton::clicked,this,&ChirpConfigWidget::removeSegments);
    connect(ui->clearButton,&QPushButton::clicked,this,&ChirpConfigWidget::clear);
    connect(ui->loadButton,&QPushButton::clicked,this,&ChirpConfigWidget::load);

    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    connect(ui->preChirpProtectionDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->preChirpDelayDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->postChirpDelayDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->postChirpProtectionDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->chirpsSpinBox,vc,p_ctm,&ChirpTableModel::setNumChirps);
    connect(ui->chirpsSpinBox,vc,ui->currentChirpBox,&QSpinBox::setMaximum);
    connect(ui->chirpsSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->chirpsSpinBox,vc,[=](int n){ui->chirpIntervalDoubleSpinBox->setEnabled(n>1);});
    connect(ui->chirpsSpinBox,vc,this,&ChirpConfigWidget::setButtonStates);
    connect(ui->chirpIntervalDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->currentChirpBox,vc,[=](int val){ p_ctm->setCurrentChirp(val-1); });
    connect(ui->applyToAllBox,&QCheckBox::toggled,p_ctm,&ChirpTableModel::setApplyToAll);

    ui->chirpTable->setItemDelegate(new ChirpDoubleSpinBoxDelegate);


}

ChirpConfigWidget::~ChirpConfigWidget()
{
    delete ui;
}

void ChirpConfigWidget::setRfConfig(const RfConfig c)
{
    if(d_rampOnly)
    {
        auto rfc = c;
        auto thiscc = rfc.getChirpConfig();
        if(!thiscc.chirpList().isEmpty())
        {
            if(thiscc.chirpList().constFirst().size() > 1)
            {
                thiscc.setChirpList(QVector<QVector<ChirpConfig::ChirpSegment>>());
                rfc.setChirpConfig(thiscc);
            }
        }
        p_ctm->setRfConfig(rfc);
    }
    else
        p_ctm->setRfConfig(c);

    clearList(false);
    auto cc = c.getChirpConfig();

    ui->preChirpProtectionDoubleSpinBox->setValue(cc.preChirpProtectionDelay());
    ui->preChirpDelayDoubleSpinBox->setValue(cc.preChirpGateDelay());
    ui->postChirpDelayDoubleSpinBox->setValue(cc.postChirpGateDelay());
    ui->postChirpProtectionDoubleSpinBox->setValue(cc.postChirpProtectionDelay());
    ui->chirpsSpinBox->setValue(cc.numChirps());
    if(!cc.chirpList().isEmpty())
    {
        ui->chirpIntervalDoubleSpinBox->setValue(cc.chirpInterval());

        p_ctm->setNumChirps(cc.numChirps());

        if(cc.allChirpsIdentical())
        {
            p_ctm->setApplyToAll(true);
            ui->applyToAllBox->setChecked(true);
            for(int j=0; j<cc.chirpList().at(0).size(); j++)
            {
                double dur = qBound(0.1,cc.segmentDuration(0,j),100000.0);

                if(cc.chirpList().at(0).at(j).empty)
                    p_ctm->addSegment(0.0,0.0,dur,p_ctm->rowCount(QModelIndex()),true);
                else
                    p_ctm->addSegment(cc.segmentStartFreq(0,j),cc.segmentEndFreq(0,j),dur,p_ctm->rowCount(QModelIndex()));
            }
        }
        else
        {
            p_ctm->setApplyToAll(false);
            ui->applyToAllBox->setChecked(false);
            ui->applyToAllBox->setEnabled(false);
            for(int i=0; i<cc.chirpList().size(); i++)
            {
                p_ctm->setCurrentChirp(i);
                for(int j=0; j<cc.chirpList().at(i).size(); j++)
                {
                    double dur = qBound(0.1,cc.segmentDuration(i,j),100000.0);

                    if(cc.chirpList().at(i).at(j).empty)
                        p_ctm->addSegment(0.0,0.0,dur,p_ctm->rowCount(QModelIndex()),true);
                    else
                        p_ctm->addSegment(cc.segmentStartFreq(i,j),cc.segmentEndFreq(i,j),dur,p_ctm->rowCount(QModelIndex()));
                }
            }
            p_ctm->setCurrentChirp(0);
        }
    }

    ui->currentChirpBox->setValue(1);
}

RfConfig ChirpConfigWidget::getRfConfig()
{
    ///TODO: Handle multiple chirp configs
    auto rfc = p_ctm->getRfConfig();

    for(int i=0; i<rfc.numChirpConfigs(); i++)
    {
        auto cc = rfc.getChirpConfig(i);
        cc.setPreChirpProtectionDelay(ui->preChirpProtectionDoubleSpinBox->value());
        cc.setPreChirpGateDelay(ui->preChirpDelayDoubleSpinBox->value()/1e3);
        cc.setPostChirpGateDelay(ui->postChirpDelayDoubleSpinBox->value()/1e3);
        cc.setPostChirpProtectionDelay(ui->postChirpProtectionDoubleSpinBox->value());
        cc.setNumChirps(ui->chirpsSpinBox->value());
        cc.setChirpInterval(ui->chirpIntervalDoubleSpinBox->value());
        rfc.setChirpConfig(cc,i);
    }

    return rfc;
}

QSpinBox *ChirpConfigWidget::numChirpsBox() const
{
    return ui->chirpsSpinBox;
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

    if(d_rampOnly)
    {
        ui->addButton->setEnabled(p_ctm->rowCount(QModelIndex()) == 0);
        ui->addEmptyButton->setEnabled(false);
    }

    //insert button only enabled if one item is selected
    ui->insertButton->setEnabled(l.size() == 1 && !d_rampOnly);
    ui->insertEmptyButton->setEnabled(l.size() == 1 && !d_rampOnly);

    //remove button active if one or more rows selected
    ui->removeButton->setEnabled(l.size() > 0);

    //move buttons enabled only if selection is contiguous
    ui->moveDownButton->setEnabled(c && p_ctm->rowCount(QModelIndex()) > 0);
    ui->moveUpButton->setEnabled(c && p_ctm->rowCount(QModelIndex()) > 0);

    //clear button only enabled if table is not empty
    ui->clearButton->setEnabled(p_ctm->rowCount(QModelIndex()) > 0);

    //get number of chirps associated with current chirp config
    ///TODO: Handle >1 CC
    auto cc = getRfConfig().getChirpConfig();
    ui->currentChirpBox->setRange(1,qMax(1,cc.numChirps()));
    ui->applyToAllBox->setEnabled(cc.allChirpsIdentical());
}

void ChirpConfigWidget::addSegment()
{
    p_ctm->addSegment(-1.0,-1.0,0.500,-1);
    updateChirpPlot();
}

void ChirpConfigWidget::addEmptySegment()
{
    p_ctm->addSegment(0.0,0.0,0.500,-1,true);
    updateChirpPlot();
}

void ChirpConfigWidget::insertSegment()
{
    QModelIndexList l = ui->chirpTable->selectionModel()->selectedRows();
    if(l.size() != 1)
        return;

    p_ctm->addSegment(-1.0,-1.0,0.500,l.at(0).row());
    updateChirpPlot();
}

void ChirpConfigWidget::insertEmptySegment()
{
    QModelIndexList l = ui->chirpTable->selectionModel()->selectedRows();
    if(l.size() != 1)
        return;

    p_ctm->addSegment(0.0,0.0,0.500,l.at(0).row(),true);
    updateChirpPlot();
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
    updateChirpPlot();
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
    updateChirpPlot();
}

void ChirpConfigWidget::clear()
{
    int ret = QMessageBox::question(this,QString("Clear all rows"),
                          QString("Are you sure you want to remove all segments from the table?"),
                          QMessageBox::Yes|QMessageBox::Cancel, QMessageBox::Cancel);

    if(ret == QMessageBox::Yes)
        clearList();
}

void ChirpConfigWidget::load()
{
    SettingsStorage s;
    int e = s.get<int>(BC::Key::exptNum,0);
    if(e < 1)
    {
        QMessageBox::critical(this,QString("Cannot Load Chirp"),QString("Cannot load chirp because no experiments have been performed."),QMessageBox::Ok);
        return;
    }
    bool ok;
    int num = QInputDialog::getInt(this,QString("Load Chirp"),QString("Load chirp from experiment"),e,1,e,1,&ok);
    if(!ok || num <= 0 || num > e)
        return;

    ///TODO: handle case when experiment has multiple chirps (e.g., some kind of DR)
    ChirpConfig cc(num);
    if(cc.chirpList().isEmpty())
    {
        QMessageBox::critical(this,QString("Load Failed"),QString("Could not open chirp from experiment %1.").arg(num));
        return;
    }

    //use chirp
    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    disconnect(ui->preChirpProtectionDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(ui->preChirpDelayDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(ui->postChirpDelayDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(ui->postChirpProtectionDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(ui->chirpsSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(ui->chirpIntervalDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::updateChirpPlot);

    p_ctm->setApplyToAll(true);
    clearList(false);

    ui->preChirpDelayDoubleSpinBox->setValue(qRound(cc.preChirpGateDelay()));
    ui->preChirpProtectionDoubleSpinBox->setValue(qRound(cc.preChirpProtectionDelay()));
    ui->postChirpDelayDoubleSpinBox->setValue(qRound(cc.postChirpGateDelay()));
    ui->postChirpProtectionDoubleSpinBox->setValue(qRound(cc.postChirpProtectionDelay()));
    ui->chirpsSpinBox->setValue(cc.numChirps());
    ui->chirpIntervalDoubleSpinBox->setValue(cc.chirpInterval());
    p_ctm->setNumChirps(cc.numChirps());
    p_ctm->setCurrentChirp(0);
    if(cc.allChirpsIdentical())
    {
        for(int i=0; i<cc.chirpList().at(0).size(); i++)
        {
            ui->applyToAllBox->setEnabled(true);
            ui->applyToAllBox->setChecked(true);
            p_ctm->setApplyToAll(true);
            p_ctm->addSegment(cc.segmentStartFreq(0,i),cc.segmentEndFreq(0,i),cc.segmentDuration(0,i),-1,cc.segmentEmpty(0,i));
        }
    }
    else
    {
        ui->applyToAllBox->setEnabled(false);
        ui->applyToAllBox->setChecked(false);
        p_ctm->setApplyToAll(false);
        for(int j=0; j<cc.chirpList().size(); j++)
        {
            p_ctm->setCurrentChirp(j);
            for(int i=0; i<cc.chirpList().at(j).size(); i++)
                p_ctm->addSegment(cc.segmentStartFreq(j,i),cc.segmentEndFreq(j,i),cc.segmentDuration(j,i),-1,cc.segmentEmpty(j,i));
        }
        p_ctm->setCurrentChirp(0);
        ui->currentChirpBox->setValue(1);
    }


    connect(ui->preChirpProtectionDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(ui->preChirpDelayDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(ui->postChirpDelayDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(ui->postChirpProtectionDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(ui->chirpsSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(ui->chirpIntervalDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);

    ///TODO: deal with return value, handle >1 CC
    auto rfc = getRfConfig();
    emit chirpConfigChanged();
    ui->chirpPlot->newChirp(rfc.getChirpConfig());
    setButtonStates();

}

void ChirpConfigWidget::updateChirpPlot()
{
    auto rfc =getRfConfig();

    emit chirpConfigChanged();
    ui->chirpPlot->newChirp(rfc.getChirpConfig());
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

void ChirpConfigWidget::clearList(bool replot)
{
    if(p_ctm->rowCount(QModelIndex()) > 0)
        p_ctm->removeRows(0,p_ctm->rowCount(QModelIndex()),QModelIndex());

    if(replot)
        updateChirpPlot();
}
