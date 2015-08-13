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

    connect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::setButtonStates);
    connect(ui->chirpTable->selectionModel(),&QItemSelectionModel::selectionChanged,this,&ChirpConfigWidget::setButtonStates);
    connect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::updateChirpPlot);
//    connect(ui->chirpTable->selectionModel(),&QItemSelectionModel::selectionChanged,this,&ChirpConfigWidget::checkChirp);

    initializeFromSettings();
    setButtonStates();

    connect(ui->addButton,&QPushButton::clicked,this,&ChirpConfigWidget::addSegment);
    connect(ui->insertButton,&QPushButton::clicked,this,&ChirpConfigWidget::insertSegment);
    connect(ui->moveUpButton,&QPushButton::clicked,[=](){ moveSegments(-1); });
    connect(ui->moveDownButton,&QPushButton::clicked,[=](){ moveSegments(1); });
    connect(ui->removeButton,&QPushButton::clicked,this,&ChirpConfigWidget::removeSegments);
    connect(ui->clearButton,&QPushButton::clicked,this,&ChirpConfigWidget::clear);
    connect(ui->loadButton,&QPushButton::clicked,this,&ChirpConfigWidget::load);

    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    connect(ui->preChirpProtectionSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->preChirpDelaySpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->postChirpProtectionSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->chirpsSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->chirpsSpinBox,vc,[=](int n){ui->chirpIntervalDoubleSpinBox->setEnabled(n>1);});
    connect(ui->chirpIntervalDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);

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

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("chirpConfig"));

    ui->preChirpProtectionSpinBox->setValue(s.value(QString("preChirpProtection"),10).toInt());
    ui->preChirpDelaySpinBox->setValue(s.value(QString("preChirpDelay"),300).toInt());
    ui->postChirpProtectionSpinBox->setValue(s.value(QString("postChirpProtection"),300).toInt());
    ui->chirpsSpinBox->setValue(s.value(QString("numChirps"),10).toInt());
    ui->chirpIntervalDoubleSpinBox->setValue(s.value(QString("chirpInterval"),20.0).toDouble());

    double chirpMin = s.value(QString("chirpMin"),26500.0).toDouble();
    double chirpMax = s.value(QString("chirpMax"),40000.0).toDouble();

    clearList();

    int numSegments = s.beginReadArray(QString("segments"));
    for(int i=0; i<numSegments; i++)
    {
        s.setArrayIndex(i);
        double start = qBound(chirpMin,s.value(QString("startFreq"),chirpMin).toDouble(),chirpMax);
        double end = qBound(chirpMin,s.value(QString("endFreq"),chirpMax).toDouble(),chirpMax);
        double dur = qBound(0.1,s.value(QString("duration"),500.0).toDouble(),100000.0);
        p_ctm->addSegment(start,end,dur,p_ctm->rowCount(QModelIndex()));
    }
    s.endArray();
    s.endGroup();

}

void ChirpConfigWidget::saveToSettings() const
{
    if(!d_currentChirpConfig.isValid())
        return;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("chirpConfig"));

    s.setValue(QString("preChirpProtection"),ui->preChirpProtectionSpinBox->value());
    s.setValue(QString("preChirpDelay"),ui->preChirpDelaySpinBox->value());
    s.setValue(QString("postChirpProtection"),ui->postChirpProtectionSpinBox->value());
    s.setValue(QString("numChirps"),ui->chirpsSpinBox->value());
    s.setValue(QString("chirpInterval"),ui->chirpIntervalDoubleSpinBox->value());

    s.beginWriteArray(QString("segments"));
    for(int i=0; i<p_ctm->segmentList().size(); i++)
    {
        s.setArrayIndex(i);
        s.setValue(QString("startFreq"),p_ctm->segmentList().at(i).startFreqMHz);
        s.setValue(QString("endFreq"),p_ctm->segmentList().at(i).endFreqMHz);
        s.setValue(QString("duration"),p_ctm->segmentList().at(i).durationUs);
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
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("chirpConfig"));
    double chirpMin = s.value(QString("chirpMin"),26500.0).toDouble();
    double chirpMax = s.value(QString("chirpMax"),40000.0).toDouble();
    s.endGroup();

    p_ctm->addSegment(chirpMin,chirpMax,0.500,-1);
    updateChirpPlot();
}

void ChirpConfigWidget::insertSegment()
{
    QModelIndexList l = ui->chirpTable->selectionModel()->selectedRows();
    if(l.size() != 1)
        return;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("chirpConfig"));
    double chirpMin = s.value(QString("chirpMin"),26500.0).toDouble();
    double chirpMax = s.value(QString("chirpMax"),40000.0).toDouble();
    s.endGroup();

    p_ctm->addSegment(chirpMin,chirpMax,0.500,l.at(0).row());
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
        QMessageBox::critical(this,QString("Load Failed"),QString("Could not open chirp from experiment %1.").arg(e));
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
    disconnect(ui->postChirpProtectionSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(ui->chirpsSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(ui->chirpIntervalDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    disconnect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::updateChirpPlot);

    clearList();
    ui->preChirpDelaySpinBox->setValue(qRound(cc.preChirpDelay()*1e3));
    ui->preChirpProtectionSpinBox->setValue(qRound(cc.preChirpProtection()*1e3));
    ui->postChirpProtectionSpinBox->setValue(qRound(cc.postChirpProtection()*1e3));
    ui->chirpsSpinBox->setValue(cc.numChirps());
    ui->chirpIntervalDoubleSpinBox->setValue(cc.chirpInterval());
    for(int i=0; i<cc.segmentList().size(); i++)
        p_ctm->addSegment(cc.segmentStartFreq(i),cc.segmentEndFreq(i),cc.segmentDuration(i),-1);


    connect(ui->preChirpProtectionSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(ui->preChirpDelaySpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(ui->postChirpProtectionSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(ui->chirpsSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(ui->chirpIntervalDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);
    connect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::updateChirpPlot,Qt::UniqueConnection);

    d_currentChirpConfig = cc;
    emit chirpConfigChanged();
    ui->chirpPlot->newChirp(d_currentChirpConfig);

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

void ChirpConfigWidget::clearList()
{
    if(p_ctm->rowCount(QModelIndex()) > 0)
        p_ctm->removeRows(0,p_ctm->rowCount(QModelIndex()),QModelIndex());

    updateChirpPlot();
}

void ChirpConfigWidget::updateChirpConfig()
{
    d_currentChirpConfig.setPreChirpProtection(ui->preChirpProtectionSpinBox->value()/1e3);
    d_currentChirpConfig.setPreChirpDelay(ui->preChirpDelaySpinBox->value()/1e3);
    d_currentChirpConfig.setPostChirpProtection(ui->postChirpProtectionSpinBox->value()/1e3);
    d_currentChirpConfig.setNumChirps(ui->chirpsSpinBox->value());
    d_currentChirpConfig.setChirpInterval(ui->chirpIntervalDoubleSpinBox->value());
    d_currentChirpConfig.setSegmentList(p_ctm->segmentList());

    emit chirpConfigChanged();

}
