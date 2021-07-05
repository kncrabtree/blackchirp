#include <gui/widget/rfconfigwidget.h>
#include "ui_rfconfigwidget.h"

RfConfigWidget::RfConfigWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::RfConfigWidget)
{
    ui->setupUi(this);

    ui->upconversionSidebandComboBox->addItem(QString("Upper"),RfConfig::UpperSideband);
    ui->downconversionSidebandComboBox->addItem(QString("Upper"),RfConfig::UpperSideband);
    ui->upconversionSidebandComboBox->addItem(QString("Lower"),RfConfig::LowerSideband);
    ui->downconversionSidebandComboBox->addItem(QString("Lower"),RfConfig::LowerSideband);

    p_ctm = new ClockTableModel();
    ui->clockTableView->setModel(p_ctm);
    ui->clockTableView->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->clockTableView->setItemDelegate(new ClockTableDelegate);

    connect(ui->commonLoCheckBox,&QCheckBox::toggled,p_ctm,&ClockTableModel::setCommonLo);

}

RfConfigWidget::~RfConfigWidget()
{
    delete ui;
}

void RfConfigWidget::setRfConfig(const RfConfig c)
{
    ui->awgMultBox->setValue(qRound(c.d_awgMult));

    if(c.d_upMixSideband == RfConfig::UpperSideband)
        ui->upconversionSidebandComboBox->setCurrentIndex(0);
    else
        ui->upconversionSidebandComboBox->setCurrentIndex(1);
    if(c.d_downMixSideband == RfConfig::UpperSideband)
        ui->downconversionSidebandComboBox->setCurrentIndex(0);
    else
        ui->downconversionSidebandComboBox->setCurrentIndex(1);

    ui->chirpMultiplicationSpinBox->setValue(qRound(c.d_chirpMult));

    ui->commonLoCheckBox->blockSignals(true);
    ui->commonLoCheckBox->setChecked(c.d_commonUpDownLO);
    ui->commonLoCheckBox->blockSignals(false);

    p_ctm->setConfig(c);
    ui->clockTableView->resizeColumnsToContents();
}

RfConfig RfConfigWidget::getRfConfig()
{
    auto rfc = p_ctm->getRfConfig();

    rfc.d_awgMult = static_cast<double>(ui->awgMultBox->value());
    rfc.d_upMixSideband = (ui->upconversionSidebandComboBox->currentData().value<RfConfig::Sideband>());
    rfc.d_chirpMult = static_cast<double>(ui->chirpMultiplicationSpinBox->value());
    rfc.d_downMixSideband = ui->downconversionSidebandComboBox->currentData().value<RfConfig::Sideband>();
    rfc.d_commonUpDownLO = ui->commonLoCheckBox->isChecked();

    return rfc;
}
