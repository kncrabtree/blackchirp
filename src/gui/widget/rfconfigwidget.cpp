#include <gui/widget/rfconfigwidget.h>
#include "ui_rfconfigwidget.h"

RfConfigWidget::RfConfigWidget(QWidget *parent) :
    QWidget(parent), SettingsStorage(BC::Key::RfConfigWidget::key),
    ui(new Ui::RfConfigWidget)
{
    using namespace BC::Key::RfConfigWidget;
    ui->setupUi(this);

    ui->upconversionSidebandComboBox->addItem(QString("Upper"),RfConfig::UpperSideband);
    ui->downconversionSidebandComboBox->addItem(QString("Upper"),RfConfig::UpperSideband);
    ui->upconversionSidebandComboBox->addItem(QString("Lower"),RfConfig::LowerSideband);
    ui->downconversionSidebandComboBox->addItem(QString("Lower"),RfConfig::LowerSideband);

    ui->upconversionSidebandComboBox->setCurrentIndex(ui->upconversionSidebandComboBox->findData(get(upSB,RfConfig::UpperSideband)));
    ui->downconversionSidebandComboBox->setCurrentIndex(ui->downconversionSidebandComboBox->findData(get(downSB,RfConfig::UpperSideband)));

    ui->awgMultBox->setValue(get(awgM,1));
    ui->chirpMultiplicationSpinBox->setValue(get(chirpM,1));
    ui->commonLoCheckBox->setChecked(get(comLO,false));

    registerGetter(upSB,std::function<QVariant()>{[this](){ return ui->upconversionSidebandComboBox->currentData(); }});
    registerGetter(downSB,std::function<QVariant()>{[this](){ return ui->downconversionSidebandComboBox->currentData(); }});

    registerGetter(awgM,ui->awgMultBox,&QSpinBox::value);
    registerGetter(chirpM,ui->chirpMultiplicationSpinBox,&QSpinBox::value);
    registerGetter<QAbstractButton>(comLO,ui->commonLoCheckBox,&QCheckBox::isChecked);

    p_ctm = new ClockTableModel(this);
    ui->clockTableView->setModel(p_ctm);
    ui->clockTableView->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->clockTableView->setItemDelegate(new ClockTableDelegate);

    connect(ui->commonLoCheckBox,&QCheckBox::toggled,p_ctm,&ClockTableModel::setCommonLo);
    connect(ui->commonLoCheckBox,&QCheckBox::toggled,this,&RfConfigWidget::edited);
    connect(p_ctm,&ClockTableModel::dataChanged,this,&RfConfigWidget::edited);

}

RfConfigWidget::~RfConfigWidget()
{
    clearGetters();
    delete ui;
}

void RfConfigWidget::setFromRfConfig(const RfConfig &c)
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

    setClocks(c);
    ui->clockTableView->resizeColumnsToContents();
}

void RfConfigWidget::setClocks(const RfConfig &c)
{
    p_ctm->setFromConfig(c);
}

void RfConfigWidget::toRfConfig(RfConfig &c)
{
    p_ctm->toRfConfig(c);

    c.d_awgMult = static_cast<double>(ui->awgMultBox->value());
    c.d_upMixSideband = (ui->upconversionSidebandComboBox->currentData().value<RfConfig::Sideband>());
    c.d_chirpMult = static_cast<double>(ui->chirpMultiplicationSpinBox->value());
    c.d_downMixSideband = ui->downconversionSidebandComboBox->currentData().value<RfConfig::Sideband>();
    c.d_commonUpDownLO = ui->commonLoCheckBox->isChecked();
}

QString RfConfigWidget::getHwKey(RfConfig::ClockType type) const
{
    return p_ctm->getHwKey(type);
}

bool RfConfigWidget::commonLO() const
{
    return ui->commonLoCheckBox->isChecked();
}
