#include "lifprocessingwidget.h"

#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QToolButton>
#include <QCheckBox>
#include <QTableWidget>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QResizeEvent>

#include <gui/style/themecolors.h>
#include <gui/widget/settingstable.h>

using namespace Qt::StringLiterals;

LifProcessingWidget::LifProcessingWidget(bool store, QWidget *parent)
    : QWidget{parent}, SettingsStorage(BC::Key::LifProcessing::key)
{
    using namespace BC::Key::LifProcessing;

    auto tt = QString("Gate position in units of points. Hold Ctrl to adjust in steps of 10");
    p_lgStartBox = new QSpinBox(this);
    p_lgStartBox->setToolTip(tt);
    p_lgStartBox->setRange(0,1000000000);
    p_lgStartBox->setValue(get(lgStart,0));

    p_lgEndBox = new QSpinBox(this);
    p_lgEndBox->setToolTip(tt);
    p_lgEndBox->setRange(1,1000000000);
    p_lgEndBox->setValue(get(lgEnd,1));

    p_rgStartBox = new QSpinBox(this);
    p_rgStartBox->setToolTip(tt);
    p_rgStartBox->setRange(0,1000000000);
    p_rgStartBox->setValue(get(rgStart,0));

    p_rgEndBox = new QSpinBox(this);
    p_rgEndBox->setToolTip(tt);
    p_rgEndBox->setRange(1,1000000000);
    p_rgEndBox->setValue(get(rgEnd,1));

    // The "Gates" heading is a SettingsTable section band so it matches
    // the Low Pass Filter / Savitzky-Golay headings below; the gate
    // values stay a true LIF/Reference x Start/End matrix (a QTableWidget
    // restyled so the row names sit in a regular first column rather than
    // the vertical header, consistent with the other restyled matrix
    // tables).
    auto gateHeader = new SettingsTable(this);
    gateHeader->setFocusPolicy(Qt::NoFocus);
    gateHeader->addSectionRow("Gates"_L1);

    auto gateTable = new QTableWidget(2,3,this);
    gateTable->setHorizontalHeaderLabels({"", "Start", "End"});
    gateTable->horizontalHeader()->setSectionResizeMode(0,QHeaderView::ResizeToContents);
    gateTable->horizontalHeader()->setSectionResizeMode(1,QHeaderView::Stretch);
    gateTable->horizontalHeader()->setSectionResizeMode(2,QHeaderView::Stretch);
    gateTable->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    gateTable->verticalHeader()->setVisible(false);
    gateTable->setSelectionMode(QAbstractItemView::NoSelection);
    gateTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    gateTable->setFocusPolicy(Qt::NoFocus);
    // Frameless to match the SettingsTable bands it sits between.
    gateTable->setFrameShape(QFrame::NoFrame);
    // Content-sized like SettingsTable: never grow past the two rows or
    // show a vertical scrollbar, so the panel does not waste height.
    gateTable->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
    gateTable->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    gateTable->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Maximum);

    auto setGateRowLabel = [gateTable](int row, const QString &text) {
        auto *it = new QTableWidgetItem(text);
        it->setFlags(Qt::ItemIsEnabled);
        gateTable->setItem(row, 0, it);
    };
    setGateRowLabel(0, "LIF"_L1);
    setGateRowLabel(1, "Reference"_L1);
    gateTable->setCellWidget(0,1,p_lgStartBox);
    gateTable->setCellWidget(0,2,p_lgEndBox);
    gateTable->setCellWidget(1,1,p_rgStartBox);
    gateTable->setCellWidget(1,2,p_rgEndBox);

    // Header band sits flush on the matrix (no inter-widget gap), the
    // way a SettingsTable section row abuts its bound rows.
    auto gateLayout = new QVBoxLayout;
    gateLayout->setContentsMargins(0,0,0,0);
    gateLayout->setSpacing(0);
    gateLayout->addWidget(gateHeader);
    gateLayout->addWidget(gateTable);

    p_lpAlphaBox = new QDoubleSpinBox(this);
    p_lpAlphaBox->setDecimals(4);
    p_lpAlphaBox->setRange(0.0,0.9999);
    p_lpAlphaBox->setSingleStep(0.01);
    p_lpAlphaBox->setSpecialValueText(QString("Disabled"));
    p_lpAlphaBox->setToolTip("Low pass filter: x_n = alpha*x_{n-1} + (1-alpha)*x_n");
    p_lpAlphaBox->setValue(get(lpAlpha,0.0));

    p_sgWinBox = new QSpinBox(this);
    p_sgWinBox->setToolTip("Savitzky-Golay window size. Must be odd");
    p_sgWinBox->setMinimum(3);
    p_sgWinBox->setSingleStep(2);

    p_sgPolyBox = new QSpinBox(this);
    p_sgPolyBox->setToolTip("Savitzky-Golay polynomial order. Must be between 2 and window size - 1");
    p_sgPolyBox->setMinimum(2);

    // Low-pass and Savitzky-Golay groups become sections of one
    // SettingsTable; the checkable Sav-Gol QGroupBox is now a checkable
    // section row whose bound rows collapse when it is unchecked.
    auto procTable = new SettingsTable(this);
    procTable->setFocusPolicy(Qt::NoFocus);
    procTable->addSectionRow("Low Pass Filter"_L1);
    procTable->addSettingRow(u"α"_s,p_lpAlphaBox,
        "Low pass filter: x_n = alpha*x_{n-1} + (1-alpha)*x_n"_L1);
    int sgSection = procTable->addCheckableSectionRow(
        "Savitzky-Golay Smoothing"_L1,get(sgEn,false),&p_sgBox);
    p_sgBox->setToolTip("Enable/disable Savitzky-Golay smoothing"_L1);
    int sgWinRow = procTable->addSettingRow("Window"_L1,p_sgWinBox,
        "Savitzky-Golay window size. Must be odd"_L1);
    int sgPolyRow = procTable->addSettingRow("Order"_L1,p_sgPolyBox,
        "Savitzky-Golay polynomial order. Must be between 2 and window size - 1"_L1);
    procTable->bindSectionRows(sgSection,{sgWinRow,sgPolyRow});

    // Text-beside-icon tool buttons that collapse to icon-only when the
    // row is too narrow (tooltips then carry the meaning), mirroring
    // OverlayManagerWidget. Reset/Save reuse the FtmwProcessingPanel
    // icons for cross-panel consistency.
    p_reprocessButton = new QToolButton(this);
    p_reprocessButton->setIcon(ThemeColors::createThemedIcon(":/icons/calculator.svg",ThemeColors::IconSecondary,this));
    p_reprocessButton->setText("Reprocess All"_L1);
    p_reprocessButton->setToolTip("Reprocess all data with the current settings."_L1);
    p_reprocessButton->setEnabled(false);
    connect(p_reprocessButton,&QToolButton::clicked,this,&LifProcessingWidget::reprocessSignal);

    p_resetButton = new QToolButton(this);
    p_resetButton->setIcon(ThemeColors::createThemedIcon(":/icons/arrow-path.svg",ThemeColors::IconSecondary,this));
    p_resetButton->setText("Reset"_L1);
    p_resetButton->setToolTip("Reset to most recently saved values."_L1);
    p_resetButton->setEnabled(false);
    connect(p_resetButton,&QToolButton::clicked,this,&LifProcessingWidget::resetSignal);

    p_saveButton = new QToolButton(this);
    p_saveButton->setIcon(ThemeColors::createThemedIcon(":/icons/arrow-down-tray.svg",ThemeColors::IconSecondary,this));
    p_saveButton->setText("Save"_L1);
    p_saveButton->setToolTip("Save the current values. They will be the new defaults if this experiment is viewed again."_L1);
    p_saveButton->setEnabled(false);
    connect(p_saveButton,&QToolButton::clicked,this,&LifProcessingWidget::saveSignal);

    p_btnLayout = new QHBoxLayout;
    p_btnLayout->addWidget(p_reprocessButton,1);
    p_btnLayout->addWidget(p_resetButton,1);
    p_btnLayout->addWidget(p_saveButton,1);

    auto vbl = new QVBoxLayout;
    vbl->addLayout(gateLayout);
    vbl->addWidget(procTable);
    vbl->addLayout(p_btnLayout);
    vbl->addStretch(1);
    setLayout(vbl);
    adjustButtonStyle();

    connect(p_lgStartBox,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){
        auto v = p_lgEndBox->value();
        if(n >= v)
            p_lgEndBox->setValue(n+1);
    });
    connect(p_lgEndBox,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){
        auto v = p_lgStartBox->value();
        if(v >= n)
            p_lgStartBox->setValue(n-1);
    });
    connect(p_rgStartBox,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){
        auto v = p_rgEndBox->value();
        if(n >= v)
            p_rgEndBox->setValue(n+1);
    });
    connect(p_rgEndBox,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){
        auto v = p_rgStartBox->value();
        if(v >= n)
            p_rgStartBox->setValue(n-1);
    });
    connect(p_sgWinBox,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){
        if(!(n%2))
        {
            n--;
            p_sgWinBox->blockSignals(true);
            p_sgWinBox->setValue(n);
            p_sgWinBox->blockSignals(false);
        }

        p_sgPolyBox->setMaximum(n-1);
    });

    connect(p_lgStartBox,qOverload<int>(&QSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);
    connect(p_lgEndBox,qOverload<int>(&QSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);
    connect(p_rgStartBox,qOverload<int>(&QSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);
    connect(p_rgEndBox,qOverload<int>(&QSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);
    connect(p_lpAlphaBox,qOverload<double>(&QDoubleSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);
    connect(p_sgBox,&QCheckBox::toggled,this,&LifProcessingWidget::settingChanged);
    connect(p_sgWinBox,qOverload<int>(&QSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);
    connect(p_sgPolyBox,qOverload<int>(&QSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);

    p_sgWinBox->setValue(get(sgWin,11));
    p_sgPolyBox->setValue(get(sgPoly,3));

    if(store)
    {
        registerGetter(lgStart,p_lgStartBox,&QSpinBox::value);
        registerGetter(lgEnd,p_lgEndBox,&QSpinBox::value);
        registerGetter(rgStart,p_rgStartBox,&QSpinBox::value);
        registerGetter(rgEnd,p_rgEndBox,&QSpinBox::value);
        registerGetter(lpAlpha,p_lpAlphaBox,&QDoubleSpinBox::value);
        registerGetter(sgEn,static_cast<QAbstractButton*>(p_sgBox),&QCheckBox::isChecked);
        registerGetter(sgWin,p_sgWinBox,&QSpinBox::value);
        registerGetter(sgPoly,p_sgPolyBox,&QSpinBox::value);
    }
}

void LifProcessingWidget::initialize(int recLen, bool ref)
{
    p_lgStartBox->setRange(0,recLen-2);
    p_lgEndBox->setRange(1,recLen-1);
    p_rgStartBox->setRange(0,recLen-2);
    p_rgEndBox->setRange(1,recLen-1);

    p_rgStartBox->setEnabled(ref);
    p_rgEndBox->setEnabled(ref);
}

void LifProcessingWidget::setAll(const LifTrace::LifProcSettings &lc)
{
    blockSignals(true);
    p_lgStartBox->setValue(lc.lifGateStart);
    p_lgEndBox->setValue(lc.lifGateEnd);
    p_rgStartBox->setValue(lc.refGateStart);
    p_rgEndBox->setValue(lc.refGateEnd);
    p_lpAlphaBox->setValue(lc.lowPassAlpha);
    p_sgBox->setChecked(lc.savGolEnabled);
    p_sgWinBox->setValue(lc.savGolWin);
    p_sgPolyBox->setValue(lc.savGolPoly);
    blockSignals(false);
    emit settingChanged();
}

LifTrace::LifProcSettings LifProcessingWidget::getSettings() const
{
    return {p_lgStartBox->value(),
                p_lgEndBox->value(),
                p_rgStartBox->value(),
                p_rgEndBox->value(),
                p_lpAlphaBox->value(),
                p_sgBox->isChecked(),
                p_sgWinBox->value(),
                p_sgPolyBox->value()
    };
}

void LifProcessingWidget::experimentComplete()
{
    setEnabled(true);
    p_reprocessButton->setEnabled(true);
    p_resetButton->setEnabled(true);
    p_saveButton->setEnabled(true);
}

void LifProcessingWidget::adjustButtonStyle()
{
    // Measure with labels shown; if the row would overflow the widget
    // width, fall back to icon-only (tooltips already describe each).
    for(auto *b : {p_reprocessButton,p_resetButton,p_saveButton})
        b->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

    const auto style = (p_btnLayout->sizeHint().width() <= width())
                           ? Qt::ToolButtonTextBesideIcon
                           : Qt::ToolButtonIconOnly;
    for(auto *b : {p_reprocessButton,p_resetButton,p_saveButton})
        b->setToolButtonStyle(style);
}

void LifProcessingWidget::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    adjustButtonStyle();
}
