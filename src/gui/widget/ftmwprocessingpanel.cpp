#include <gui/widget/ftmwprocessingpanel.h>

#include <QVBoxLayout>
#include <QGridLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QPushButton>
#include <QLineEdit>
#include <QMetaEnum>

#include <gui/style/themecolors.h>
#include <gui/widget/cellwidgethelpers.h>

using BC::Gui::centerCellWidget;

namespace {

QComboBox *makeCenteredCombo()
{
    auto *cb = new QComboBox;
    cb->setEditable(true);
    cb->lineEdit()->setReadOnly(true);
    cb->lineEdit()->setAlignment(Qt::AlignCenter);
    return cb;
}

void recenterCombo(QComboBox *cb)
{
    for(int i=0; i<cb->count(); ++i)
        cb->setItemData(i,Qt::AlignCenter,Qt::TextAlignmentRole);
}

}

FtmwProcessingPanel::FtmwProcessingPanel(bool mainWin, QWidget *parent) :
    QWidget(parent), SettingsStorage(BC::Key::ftmwProcWidget)
{
    const QStringList labels{
        QStringLiteral("FT Start"),
        QStringLiteral("FT End"),
        QStringLiteral("Exp Filter"),
        QStringLiteral("VScale Ignore"),
        QStringLiteral("Zero Pad"),
        QStringLiteral("Remove DC"),
        QStringLiteral("Window Function"),
        QStringLiteral("FT Units")
    };

    p_table = new QTableWidget(labels.size(),1,this);
    p_table->setVerticalHeaderLabels(labels);
    p_table->horizontalHeader()->setVisible(false);
    p_table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    p_table->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    p_table->setSelectionMode(QAbstractItemView::NoSelection);
    p_table->setFocusPolicy(Qt::NoFocus);
    p_table->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    p_table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    p_startBox = new QDoubleSpinBox;
    p_startBox->setMinimum(0.0);
    p_startBox->setDecimals(4);
    p_startBox->setSingleStep(0.05);
    p_startBox->setSuffix(QString::fromUtf8(" μs"));
    p_startBox->setValue(get<double>(BC::Key::fidStart,0.0));
    p_startBox->setAlignment(Qt::AlignCenter);
    p_startBox->setKeyboardTracking(false);
    p_startBox->setToolTip(QStringLiteral("Start of data for FT. Points before this in the FID will be set to 0."));
    p_table->setCellWidget(0,0,p_startBox);
    if(mainWin)
        registerGetter(BC::Key::fidStart,p_startBox,&QDoubleSpinBox::value);

    p_endBox = new QDoubleSpinBox;
    p_endBox->setMinimum(0.0);
    p_endBox->setDecimals(4);
    p_endBox->setSingleStep(0.05);
    p_endBox->setSuffix(QString::fromUtf8(" μs"));
    p_endBox->setValue(get<double>(BC::Key::fidEnd,99.0));
    p_endBox->setAlignment(Qt::AlignCenter);
    p_endBox->setKeyboardTracking(false);
    p_endBox->setToolTip(QStringLiteral("End of data for FT. Points after this in the FID will be set to 0."));
    p_table->setCellWidget(1,0,p_endBox);
    if(mainWin)
        registerGetter(BC::Key::fidEnd,p_endBox,&QDoubleSpinBox::value);

    p_expBox = new QDoubleSpinBox;
    p_expBox->setMinimum(0.0);
    p_expBox->setDecimals(1);
    p_expBox->setSingleStep(0.1);
    p_expBox->setSpecialValueText(QStringLiteral("Disabled"));
    p_expBox->setSuffix(QString::fromUtf8(" μs"));
    p_expBox->setValue(get<double>(BC::Key::fidExp,0.0));
    p_expBox->setAlignment(Qt::AlignCenter);
    p_expBox->setKeyboardTracking(false);
    p_expBox->setToolTip(QStringLiteral("Time constant for exponential filter."));
    p_table->setCellWidget(2,0,p_expBox);
    if(mainWin)
        registerGetter(BC::Key::fidExp,p_expBox,&QDoubleSpinBox::value);

    p_autoScaleIgnoreBox = new QDoubleSpinBox;
    p_autoScaleIgnoreBox->setRange(0.0,1000.0);
    p_autoScaleIgnoreBox->setDecimals(1);
    p_autoScaleIgnoreBox->setSuffix(QStringLiteral(" MHz"));
    p_autoScaleIgnoreBox->setValue(get<double>(BC::Key::autoscaleIgnore,0.0));
    p_autoScaleIgnoreBox->setAlignment(Qt::AlignCenter);
    p_autoScaleIgnoreBox->setKeyboardTracking(false);
    p_autoScaleIgnoreBox->setToolTip(QStringLiteral("Points within this frequency of the LO are ignored when calculating the autoscale minimum and maximum."));
    p_table->setCellWidget(3,0,p_autoScaleIgnoreBox);
    if(mainWin)
        registerGetter(BC::Key::autoscaleIgnore,p_autoScaleIgnoreBox,&QDoubleSpinBox::value);

    p_zeroPadBox = new QSpinBox;
    p_zeroPadBox->setRange(0,2);
    p_zeroPadBox->setSpecialValueText(QStringLiteral("None"));
    p_zeroPadBox->setValue(get(BC::Key::zeroPad,0));
    p_zeroPadBox->setAlignment(Qt::AlignCenter);
    p_zeroPadBox->setKeyboardTracking(false);
    p_zeroPadBox->setToolTip(QStringLiteral("Pad FID with zeroes until length extends to a power of 2.\n1 = next power of 2, 2 = second power of 2, etc."));
    p_table->setCellWidget(4,0,p_zeroPadBox);
    registerGetter(BC::Key::zeroPad,p_zeroPadBox,&QSpinBox::value);

    p_removeDCBox = new QCheckBox;
    p_removeDCBox->setChecked(get(BC::Key::removeDC,false));
    p_removeDCBox->setToolTip(QStringLiteral("Subtract any DC offset in the FID."));
    centerCellWidget(p_table,5,0,p_removeDCBox);
    if(mainWin)
        registerGetter<bool>(BC::Key::removeDC,std::function<bool()>{
            [this](){ return p_removeDCBox->isChecked(); }});

    p_winfBox = makeCenteredCombo();
    {
        auto me = QMetaEnum::fromType<FtWorker::FtWindowFunction>();
        for(int i=0; i<me.keyCount(); ++i)
            p_winfBox->addItem(QString::fromLatin1(me.key(i)),
                               QVariant::fromValue<FtWorker::FtWindowFunction>(static_cast<FtWorker::FtWindowFunction>(me.value(i))));
    }
    recenterCombo(p_winfBox);
    p_winfBox->setCurrentIndex(p_winfBox->findData(QVariant::fromValue(get(BC::Key::ftWinf,FtWorker::None))));
    p_table->setCellWidget(6,0,p_winfBox);
    if(mainWin)
        registerGetter<FtWorker::FtWindowFunction>(BC::Key::ftWinf,std::function<FtWorker::FtWindowFunction()>{
            [this](){ return p_winfBox->currentData().value<FtWorker::FtWindowFunction>(); }});

    p_unitsBox = makeCenteredCombo();
    {
        auto me = QMetaEnum::fromType<FtWorker::FtUnits>();
        for(int i=0; i<me.keyCount(); ++i)
            p_unitsBox->addItem(QString::fromLatin1(me.key(i)),
                                QVariant::fromValue<FtWorker::FtUnits>(static_cast<FtWorker::FtUnits>(me.value(i))));
    }
    recenterCombo(p_unitsBox);
    p_unitsBox->setCurrentIndex(p_unitsBox->findData(QVariant::fromValue(get(BC::Key::ftUnits,FtWorker::FtuV))));
    p_table->setCellWidget(7,0,p_unitsBox);
    if(mainWin)
        registerGetter<FtWorker::FtUnits>(BC::Key::ftUnits,std::function<FtWorker::FtUnits()>{
            [this](){ return p_unitsBox->currentData().value<FtWorker::FtUnits>(); }});

    p_resetButton = new QPushButton(ThemeColors::createThemedIcon(":/icons/arrow-path.svg",ThemeColors::IconSecondary,this),
                                    QStringLiteral("Reset"));
    p_resetButton->setToolTip(QStringLiteral("Reset processing settings to last saved values."));
    p_saveButton = new QPushButton(ThemeColors::createThemedIcon(":/icons/arrow-down-tray.svg",ThemeColors::IconSecondary,this),
                                   QStringLiteral("Save"));
    p_saveButton->setToolTip(QStringLiteral("Save current processing settings."));

    auto *btnRow = new QHBoxLayout;
    btnRow->setContentsMargins(0,0,0,0);
    btnRow->addWidget(p_resetButton);
    btnRow->addWidget(p_saveButton);

    auto *outer = new QVBoxLayout;
    outer->setContentsMargins(4,4,4,4);
    outer->addWidget(p_table,1);
    outer->addLayout(btnRow,0);
    setLayout(outer);

    auto onDouble = [this](){ readSettings(); };
    auto onInt = [this](){ readSettings(); };
    connect(p_startBox, qOverload<double>(&QDoubleSpinBox::valueChanged), this, onDouble);
    connect(p_endBox, qOverload<double>(&QDoubleSpinBox::valueChanged), this, onDouble);
    connect(p_expBox, qOverload<double>(&QDoubleSpinBox::valueChanged), this, onDouble);
    connect(p_autoScaleIgnoreBox, qOverload<double>(&QDoubleSpinBox::valueChanged), this, onDouble);
    connect(p_zeroPadBox, qOverload<int>(&QSpinBox::valueChanged), this, onInt);
    connect(p_removeDCBox, &QCheckBox::toggled, this, [this](bool){ readSettings(); });
    connect(p_winfBox, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){ readSettings(); });
    connect(p_unitsBox, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){ readSettings(); });
    connect(p_resetButton, &QPushButton::clicked, this, [this](){ emit resetSignal(); });
    connect(p_saveButton, &QPushButton::clicked, this, [this](){ emit saveSignal(); });

    if(!mainWin)
        discardChanges();
}

FtmwProcessingPanel::~FtmwProcessingPanel() = default;

FtWorker::FidProcessingSettings FtmwProcessingPanel::getSettings()
{
    double start = p_startBox->value();
    double stop = p_endBox->value();
    double expf = p_expBox->value();
    bool rdc = p_removeDCBox->isChecked();
    int zeroPad = p_zeroPadBox->value();
    double ignore = p_autoScaleIgnoreBox->value();
    auto units = p_unitsBox->currentData().value<FtWorker::FtUnits>();
    auto winf = p_winfBox->currentData().value<FtWorker::FtWindowFunction>();

    save();

    return { start, stop, expf, zeroPad, rdc, units, ignore, winf };
}

void FtmwProcessingPanel::setAll(const FtWorker::FidProcessingSettings &c)
{
    auto b = signalsBlocked();
    blockSignals(true);
    p_startBox->setValue(c.startUs);
    p_endBox->setValue(c.endUs);
    p_expBox->setValue(c.expFilter);
    p_removeDCBox->setChecked(c.removeDC);
    p_zeroPadBox->setValue(c.zeroPadFactor);
    p_autoScaleIgnoreBox->setValue(c.autoScaleIgnoreMHz);
    p_unitsBox->setCurrentIndex(p_unitsBox->findData(QVariant::fromValue(c.units)));
    p_winfBox->setCurrentIndex(p_winfBox->findData(QVariant::fromValue(c.windowFunction)));
    blockSignals(b);

    emit settingsUpdated(getSettings());
}

void FtmwProcessingPanel::prepareForExperient(const Experiment &e)
{
    if(e.ftmwEnabled())
    {
        p_startBox->setRange(0.0,e.ftmwConfig()->fidDurationUs());
        p_endBox->setRange(0.0,e.ftmwConfig()->fidDurationUs());
        p_expBox->setRange(0.0,10.0*(e.ftmwConfig()->fidDurationUs()));

        p_resetButton->setEnabled(e.d_number > 0);
        p_saveButton->setEnabled(e.d_number > 0);

        FtWorker::FidProcessingSettings s;
        if(e.ftmwConfig()->storage()->readProcessingSettings(s))
            setAll(s);
        else
        {
            if(e.d_number > 0)
                e.ftmwConfig()->storage()->writeProcessingSettings(getSettings());
        }
    }

    setEnabled(e.ftmwEnabled());
}

void FtmwProcessingPanel::readSettings()
{
    if(signalsBlocked())
        return;

    p_startBox->setMaximum(p_endBox->value());
    p_endBox->setMinimum(p_startBox->value());

    emit settingsUpdated(getSettings());
}
