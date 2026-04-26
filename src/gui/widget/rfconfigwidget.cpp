#include <gui/widget/rfconfigwidget.h>
#include "ui_rfconfigwidget.h"

#include <QPushButton>

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
    auto savedComLO = get(comLO,false);

    registerGetter(upSB,std::function<QVariant()>{[this](){ return ui->upconversionSidebandComboBox->currentData(); }});
    registerGetter(downSB,std::function<QVariant()>{[this](){ return ui->downconversionSidebandComboBox->currentData(); }});

    registerGetter(awgM,ui->awgMultBox,&QSpinBox::value);
    registerGetter(chirpM,ui->chirpMultiplicationSpinBox,&QSpinBox::value);
    registerGetter<QAbstractButton>(comLO,ui->commonLoCheckBox,&QCheckBox::isChecked);

    p_ctm = new ClockTableModel(this);
    ui->clockTableView->setModel(p_ctm);
    ui->clockTableView->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    ui->clockTableView->horizontalHeader()->setStretchLastSection(true);
    ui->clockTableView->setItemDelegate(new ClockTableDelegate);

    connect(ui->commonLoCheckBox,&QCheckBox::toggled,p_ctm,&ClockTableModel::setCommonLo);
    ui->commonLoCheckBox->setChecked(savedComLO);
    connect(ui->commonLoCheckBox,&QCheckBox::toggled,this,&RfConfigWidget::edited);
    connect(p_ctm,&ClockTableModel::dataChanged,this,&RfConfigWidget::edited);

    auto applyButton = new QPushButton(tr("Apply Clock Settings Now"), this);
    applyButton->setToolTip(tr("Send current clock configuration to hardware immediately."));
    ui->clockConfigBox->layout()->addWidget(applyButton);
    connect(applyButton, &QPushButton::clicked, this, [this](){
        RfConfig rfc;
        toRfConfig(rfc);
        emit applyClocks(rfc.getClocks());
    });

}

RfConfigWidget::~RfConfigWidget()
{
    clearGetters();
    delete ui;
}

void RfConfigWidget::setFromRfConfig(const RfConfig &c)
{
    ui->awgMultBox->setValue(qRound(c.d_awgMult));

    ui->upconversionSidebandComboBox->setCurrentIndex(
        ui->upconversionSidebandComboBox->findData(static_cast<int>(c.d_upMixSideband)));
    ui->downconversionSidebandComboBox->setCurrentIndex(
        ui->downconversionSidebandComboBox->findData(static_cast<int>(c.d_downMixSideband)));

    ui->chirpMultiplicationSpinBox->setValue(qRound(c.d_chirpMult));

    ui->commonLoCheckBox->blockSignals(true);
    ui->commonLoCheckBox->setChecked(c.d_commonUpDownLO);
    ui->commonLoCheckBox->blockSignals(false);

    setClocks(c.getClocks());
}

void RfConfigWidget::setClocks(const QHash<RfConfig::ClockType, RfConfig::ClockFreq> c)
{
    p_ctm->setClocks(c);
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

QSize RfConfigWidget::sizeHint() const
{
    return {750,500};
}
