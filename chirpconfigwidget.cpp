#include "chirpconfigwidget.h"
#include "ui_chirpconfigwidget.h"

#include <QSettings>
#include <QMessageBox>
#include <QSpinBox>
#include <QInputDialog>

ChirpConfigWidget::ChirpConfigWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ChirpConfigWidget), p_ctm(new ChirpTableModel(this))
{
    ui->setupUi(this);
    ui->chirpTable->setModel(p_ctm);
    ui->chirpTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);;

    initializeFromSettings();

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
    connect(ui->preChirpProtectionSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->preChirpDelaySpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->postChirpDelaySpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->postChirpProtectionSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
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

ChirpConfig ChirpConfigWidget::getChirpConfig()
{
    return d_currentChirpConfig;
}

QSpinBox *ChirpConfigWidget::numChirpsBox() const
{
    return ui->chirpsSpinBox;
}

void ChirpConfigWidget::initializeFromSettings()
{
    d_currentChirpConfig = ChirpConfig::loadFromSettings();

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    double chirpMin = s.value(QString("rfConfig/chirpMin"),26500.0).toDouble();
    double chirpMax = s.value(QString("rfConfig/chirpMax"),40000.0).toDouble();

    s.beginGroup(QString("chirpConfig"));
    double minPreProt = s.value(QString("minPreChirpProtection"),0.010).toDouble();
    double minTwt = s.value(QString("minPreChirpDelay"),0.100).toDouble();
    double minPostTwt = s.value(QString("minPostChirpDelay"),0.0).toDouble();
    double minPostProt = s.value(QString("minPostChirpProtection"),0.100).toDouble();
    s.endGroup();

    ui->preChirpProtectionSpinBox->setMinimum(minPreProt*1000);
    ui->preChirpDelaySpinBox->setMinimum(minTwt*1000);
    ui->postChirpDelaySpinBox->setMinimum(minPostTwt*1000);
    ui->postChirpProtectionSpinBox->setMinimum(minPostProt*1000);

    if(d_currentChirpConfig.isValid())
    {
        ui->preChirpProtectionSpinBox->setValue(d_currentChirpConfig.preChirpProtection()*1000);
        ui->preChirpDelaySpinBox->setValue(d_currentChirpConfig.preChirpDelay()*1000);
        ui->postChirpDelaySpinBox->setValue(d_currentChirpConfig.postChirpDelay()*1000);
        ui->postChirpProtectionSpinBox->setValue(d_currentChirpConfig.postChirpProtection()*1000);
        ui->chirpsSpinBox->setValue(d_currentChirpConfig.numChirps());
        ui->chirpIntervalDoubleSpinBox->setValue(d_currentChirpConfig.chirpInterval());

        p_ctm->setNumChirps(d_currentChirpConfig.numChirps());

        if(d_currentChirpConfig.allChirpsIdentical())
        {
            p_ctm->setApplyToAll(true);
            ui->applyToAllBox->setChecked(true);
            for(int j=0; j<d_currentChirpConfig.chirpList().at(0).size(); j++)
            {
                double dur = qBound(0.1,d_currentChirpConfig.segmentDuration(0,j),100000.0);

                if(d_currentChirpConfig.chirpList().at(0).at(j).empty)
                    p_ctm->addSegment(0.0,0.0,dur,p_ctm->rowCount(QModelIndex()),true);
                else
                {
                    double start = qBound(chirpMin,d_currentChirpConfig.segmentStartFreq(0,j),chirpMax);
                    double end = qBound(chirpMin,d_currentChirpConfig.segmentEndFreq(0,j),chirpMax);
                    p_ctm->addSegment(start,end,dur,p_ctm->rowCount(QModelIndex()));
                }
            }
        }
        else
        {
            p_ctm->setApplyToAll(false);
            ui->applyToAllBox->setChecked(false);
            ui->applyToAllBox->setEnabled(false);
            for(int i=0; i<d_currentChirpConfig.chirpList().size(); i++)
            {
                p_ctm->setCurrentChirp(i);
                for(int j=0; j<d_currentChirpConfig.chirpList().at(i).size(); j++)
                {
                    double dur = qBound(0.1,d_currentChirpConfig.segmentDuration(i,j),100000.0);

                    if(d_currentChirpConfig.chirpList().at(i).at(j).empty)
                        p_ctm->addSegment(0.0,0.0,dur,p_ctm->rowCount(QModelIndex()),true);
                    else
                    {
                        double start = qBound(chirpMin,d_currentChirpConfig.segmentStartFreq(i,j),chirpMax);
                        double end = qBound(chirpMin,d_currentChirpConfig.segmentEndFreq(i,j),chirpMax);
                        p_ctm->addSegment(start,end,dur,p_ctm->rowCount(QModelIndex()));
                    }
                }
            }
            p_ctm->setCurrentChirp(0);
        }
    }

    ui->currentChirpBox->setValue(1);
    updateChirpPlot();

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
    ui->insertEmptyButton->setEnabled(l.size() == 1);

    //remove button active if one or more rows selected
    ui->removeButton->setEnabled(l.size() > 0);

    //move buttons enabled only if selection is contiguous
    ui->moveDownButton->setEnabled(c);
    ui->moveUpButton->setEnabled(c);

    //clear button only enabled if table is not empty
    ui->clearButton->setEnabled(p_ctm->rowCount(QModelIndex()) > 0);

    ui->currentChirpBox->setRange(1,d_currentChirpConfig.numChirps());
    ui->applyToAllBox->setEnabled(d_currentChirpConfig.allChirpsIdentical());
}

void ChirpConfigWidget::addSegment()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("rfConfig"));
    double chirpMin = s.value(QString("chirpMin"),26500.0).toDouble();
    double chirpMax = s.value(QString("chirpMax"),40000.0).toDouble();
    s.endGroup();

    p_ctm->addSegment(chirpMin,chirpMax,0.500,-1);
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

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("rfConfig"));
    double chirpMin = s.value(QString("chirpMin"),26500.0).toDouble();
    double chirpMax = s.value(QString("chirpMax"),40000.0).toDouble();
    s.endGroup();

    p_ctm->addSegment(chirpMin,chirpMax,0.500,l.at(0).row());
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
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    int e = s.value(QString("exptNum"),0).toInt();
    if(e < 1)
    {
        QMessageBox::critical(this,QString("Cannot Load Chirp"),QString("Cannot load chirp because no experiments have been performed."),QMessageBox::Ok);
        return;
    }
    bool ok;
    int num = QInputDialog::getInt(this,QString("Load Chirp"),QString("Load chirp from experiment"),e,1,e,1,&ok);
    if(!ok || num <= 0 || num > e)
        return;

    ChirpConfig cc(num);
    if(!cc.isValid())
    {
        QMessageBox::critical(this,QString("Load Failed"),QString("Could not open chirp from experiment %1.").arg(num));
        return;
    }

    //check if rf config parameters are the same...
    if(!d_currentChirpConfig.compareTxParams(cc))
    {
        QString labelTable;
        QString nl("\n");
        QString tab("\t");
        QTextStream t(&labelTable);
        t.setFieldAlignment(QTextStream::AlignLeft);
        t << QString("Setting            ") << tab << QString("Current") << tab << QString("Experiment %1").arg(e) << nl;
        t << QString("Synth TX Mult ") << tab << d_currentChirpConfig.synthTxMult() << tab << cc.synthTxMult() << nl;
        t << QString("AWG Mult          ") << tab << d_currentChirpConfig.awgMult() << tab << cc.awgMult() << nl;
        t << QString("Mixer Sideband") << tab << d_currentChirpConfig.mixerSideband() << tab << cc.mixerSideband() << nl;
        t << QString("Total Mult        ") << tab << d_currentChirpConfig.totalMult() << tab << cc.totalMult();
        t.flush();

        QMessageBox::critical(this,QString("Configuration Error"),QString("TX settings from experiment %1 do not match current settings.\nIf you wish to use these settings, make the appropriate changes in the RF configuration menu.\n\n").arg(e) + labelTable);
        return;
    }

    //warn if valon tx freq is different
    if(!qFuzzyCompare(d_currentChirpConfig.synthTxFreq(),cc.synthTxFreq()))
    {
        if(QMessageBox::question(this,QString("Change TX Freq?"),QString("The TX frequency from experiment %1 (%2) does not match the current TX frequency (%3).\nIf you continue, the TX frequency will be changed when the scan begins.\n\nDo you wish to continue?").arg(num).arg(cc.synthTxFreq()).arg(d_currentChirpConfig.synthTxFreq()),QMessageBox::Yes|QMessageBox::No,QMessageBox::No) == QMessageBox::No)
            return;
    }

    //use chirp
    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    disconnect(ui->preChirpProtectionSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(ui->preChirpDelaySpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(ui->postChirpDelaySpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(ui->postChirpProtectionSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(ui->chirpsSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(ui->chirpIntervalDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::updateChirpPlot);

    p_ctm->setApplyToAll(true);
    clearList(false);

    ui->preChirpDelaySpinBox->setValue(qRound(cc.preChirpDelay()*1e3));
    ui->preChirpProtectionSpinBox->setValue(qRound(cc.preChirpProtection()*1e3));
    ui->postChirpDelaySpinBox->setValue(qRound(cc.postChirpDelay()*1e3));
    ui->postChirpProtectionSpinBox->setValue(qRound(cc.postChirpProtection()*1e3));
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


    connect(ui->preChirpProtectionSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(ui->preChirpDelaySpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(ui->postChirpDelaySpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(ui->postChirpProtectionSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(ui->chirpsSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(ui->chirpIntervalDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);

    d_currentChirpConfig = cc;
    emit chirpConfigChanged();
    ui->chirpPlot->newChirp(d_currentChirpConfig);
    setButtonStates();

}

void ChirpConfigWidget::updateChirpPlot()
{
    updateChirpConfig();
    ui->chirpPlot->newChirp(d_currentChirpConfig);
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

void ChirpConfigWidget::updateChirpConfig()
{
    d_currentChirpConfig.setPreChirpProtection(ui->preChirpProtectionSpinBox->value()/1e3);
    d_currentChirpConfig.setPreChirpDelay(ui->preChirpDelaySpinBox->value()/1e3);
    d_currentChirpConfig.setPostChirpDelay(ui->postChirpDelaySpinBox->value()/1e3);
    d_currentChirpConfig.setPostChirpProtection(ui->postChirpProtectionSpinBox->value()/1e3);
    d_currentChirpConfig.setNumChirps(ui->chirpsSpinBox->value());
    d_currentChirpConfig.setChirpInterval(ui->chirpIntervalDoubleSpinBox->value());
    d_currentChirpConfig.setChirpList(p_ctm->chirpList());

    emit chirpConfigChanged();

}
