#include "rfconfigwidget.h"
#include "ui_rfconfigwidget.h"

RfConfigWidget::RfConfigWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::RfConfigWidget)
{
    ui->setupUi(this);

    ui->upconversionSidebandComboBox->addItem(QString("Upper"),BlackChirp::UpperSideband);
    ui->downconversionSidebandComboBox->addItem(QString("Upper"),BlackChirp::UpperSideband);
    ui->upconversionSidebandComboBox->addItem(QString("Lower"),BlackChirp::LowerSideband);
    ui->downconversionSidebandComboBox->addItem(QString("Lower"),BlackChirp::LowerSideband);

    p_ctm = new ClockTableModel();
    ui->clockTableView->setModel(p_ctm);
    ui->clockTableView->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->clockTableView->setItemDelegate(new ClockTableDelegate);

}

RfConfigWidget::~RfConfigWidget()
{
    delete ui;
}

void RfConfigWidget::setRfConfig(const RfConfig c)
{
    ui->awgMultBox->setValue(qRound(c.awgMult()));

    if(c.upMixSideband() == BlackChirp::UpperSideband)
        ui->upconversionSidebandComboBox->setCurrentIndex(0);
    else
        ui->upconversionSidebandComboBox->setCurrentIndex(1);
    if(c.downMixSideband() == BlackChirp::UpperSideband)
        ui->downconversionSidebandComboBox->setCurrentIndex(0);
    else
        ui->downconversionSidebandComboBox->setCurrentIndex(1);

    ui->chirpMultiplicationSpinBox->setValue(qRound(c.chirpMult()));

    ui->commonLoCheckBox->setChecked(c.commonLO());

    p_ctm->setConfig(c);
    ui->clockTableView->resizeColumnsToContents();
}

RfConfig RfConfigWidget::getRfConfig()
{
    auto rfc = p_ctm->getRfConfig();

    rfc.setAwgMult(static_cast<double>(ui->awgMultBox->value()));
    rfc.setUpMixSideband(static_cast<BlackChirp::Sideband>(ui->upconversionSidebandComboBox->currentData().toInt()));
    rfc.setChirpMult(static_cast<double>(ui->chirpMultiplicationSpinBox->value()));
    rfc.setDownMixSideband(static_cast<BlackChirp::Sideband>(ui->downconversionSidebandComboBox->currentData().toInt()));

    return rfc;
}
