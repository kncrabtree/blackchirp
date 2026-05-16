#include "peakfindwidget.h"
#include <gui/style/themecolors.h>
#include <gui/widget/ftmwviewwidget.h>

#include <QThread>
#include <QDialog>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QPushButton>
#include <QDialogButtonBox>
#include <QToolBar>
#include <QAction>
#include <QTableView>
#include <QHeaderView>
#include <QDockWidget>
#include <QResizeEvent>
#include <QShowEvent>
#include <QCursor>
#include <QCheckBox>
#include <QLabel>
#include <QHBoxLayout>
#include <QSignalBlocker>
#include <QtConcurrent/QtConcurrent>

#include <gui/widget/scientificspinbox.h>

#include <limits>

#include <gui/dialog/peaklistexportdialog.h>

PeakFindWidget::PeakFindWidget(Ft ft, int number, QWidget *parent):
    QWidget(parent), SettingsStorage(BC::Key::peakFind),
    d_number(number), d_busy(false), d_waiting(false)
{
    setWindowTitle(QString("Peak Finder"));
    setWindowIcon(ThemeColors::createThemedIcon(":/icons/magnifying-glass-circle.svg", ThemeColors::IconPrimary, this));

    p_pf = new PeakFinder(this);
    connect(p_pf,&PeakFinder::peakList,this,&PeakFindWidget::newPeakList);

    p_listModel = new PeakListModel(this);
    p_proxy = new PeakListFilterProxyModel(this);
    p_proxy->setSourceModel(p_listModel);
    p_proxy->setSortRole(Qt::EditRole);

    setupUI();
    p_peakListView->setModel(p_proxy);

    connect(p_peakListView->selectionModel(),&QItemSelectionModel::selectionChanged,this,&PeakFindWidget::updateRemoveButton);

    d_minFreq = get<double>(BC::Key::pfMinFreq,ft.minFreqMHz());
    d_maxFreq = get<double>(BC::Key::pfMaxFreq,ft.maxFreqMHz());
    d_snr = get<double>(BC::Key::pfSnr,5.0);
    d_winSize = get<int>(BC::Key::pfWinSize,11);
    d_polyOrder = get<int>(BC::Key::pfOrder,6);

    if(d_minFreq > ft.maxFreqMHz())
        d_minFreq = ft.minFreqMHz();
    if(d_maxFreq < d_minFreq)
        d_maxFreq = ft.maxFreqMHz();

    d_currentFt = ft;

    // Restore display-filter state. The frequency spin boxes use their
    // minimum as the "unbounded" sentinel (shown via specialValueText),
    // so a saved value equal to the minimum stays unbounded on that side.
    const double fmin = ft.minFreqMHz();
    const double fmax = ft.maxFreqMHz();
    {
        QSignalBlocker bMin(p_minFreqBox), bMax(p_maxFreqBox),
                       bInt(p_minIntBox), bView(p_inViewBox), bAct(p_filterAction);
        p_minFreqBox->setRange(fmin,fmax);
        p_minFreqBox->setValue(get<double>(BC::Key::pfFilterMinFreq,fmin));
        p_maxFreqBox->setRange(fmin,fmax);
        p_maxFreqBox->setValue(get<double>(BC::Key::pfFilterMaxFreq,fmin));
        p_minIntBox->setRange(0.0,1.0e15);
        p_minIntBox->setValue(qMax(0.0,get<double>(BC::Key::pfFilterMinInt,0.0)));

        const bool fen = get<bool>(BC::Key::pfFilterEnabled,false);
        p_filterAction->setChecked(fen);
        p_inViewBox->setChecked(get<bool>(BC::Key::pfViewSync,false));
        p_filterStrip->setVisible(fen);
    }
    applyFilters();
}

PeakFindWidget::~PeakFindWidget()
{
    // A peak-find pass runs on a pooled thread and captures `this`. The
    // dock lazily destroys this widget on experiment reload / teardown, so
    // block until any in-flight pass finishes before members are freed.
    pu_watcher->waitForFinished();
}

void PeakFindWidget::setupUI()
{
    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0,0,0,0);
    mainLayout->setSpacing(0);

    // Two rows: row 1 is the "what to show" actions (find / live /
    // appearance / filter), row 2 manages the list (options / export /
    // remove / show-parent). Splitting avoids the QToolBar overflow
    // (>>) menu burying first-class controls in a narrow dock.
    p_toolBar = new QToolBar(this);
    p_toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    p_toolBar2 = new QToolBar(this);
    p_toolBar2->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

    p_findAction = p_toolBar->addAction(ThemeColors::createThemedIcon(":/icons/magnifying-glass-circle.svg", ThemeColors::IconPrimary, this), "Find Now");
    p_findAction->setToolTip("Find peaks in the current FT now");
    connect(p_findAction,&QAction::triggered,this,&PeakFindWidget::findPeaks);

    p_liveAction = p_toolBar->addAction(ThemeColors::createThemedIcon(":/icons/arrow-path.svg", ThemeColors::IconSecondary, this), "Live Update");
    p_liveAction->setToolTip("Re-find peaks automatically whenever the FT changes");
    p_liveAction->setCheckable(true);
    connect(p_liveAction,&QAction::toggled,this,[this](bool b){ if(b) findPeaks(); });

    p_appearanceAction = p_toolBar->addAction(ThemeColors::createThemedIcon(":/icons/swatch.svg", ThemeColors::IconSecondary, this), "Appearance");
    p_appearanceAction->setToolTip("Edit the peak marker appearance");
    connect(p_appearanceAction,&QAction::triggered,this,[this](){
        // Anchor the appearance menu at the button's lower-left so it
        // drops directly beneath the Appearance toolbar button.
        QWidget *w = p_toolBar->widgetForAction(p_appearanceAction);
        QPoint pos = w ? w->mapToGlobal(QPoint(0,w->height())) : QCursor::pos();
        emit editPeakAppearanceRequested(pos);
    });

    p_filterAction = p_toolBar->addAction(ThemeColors::createThemedIcon(":/icons/funnel.svg", ThemeColors::IconSecondary, this), "Filter");
    p_filterAction->setToolTip("Show the display-filter controls (frequency range, minimum intensity, in-view)");
    p_filterAction->setCheckable(true);
    connect(p_filterAction,&QAction::toggled,this,[this](bool b){
        p_filterStrip->setVisible(b);
        applyFilters();
        persistFilterState();
        adjustToolbarStyle();
    });

    auto *spacer1 = new QWidget(this);
    spacer1->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    p_toolBar->addWidget(spacer1);

    p_optionsAction = p_toolBar2->addAction(ThemeColors::createThemedIcon(":/icons/cog-6-tooth.svg", ThemeColors::IconSecondary, this), "Options...");
    p_optionsAction->setToolTip("Configure peak-finding parameters");
    connect(p_optionsAction,&QAction::triggered,this,&PeakFindWidget::launchOptionsDialog);

    p_exportAction = p_toolBar2->addAction(ThemeColors::createThemedIcon(":/icons/arrow-down-tray.svg", ThemeColors::IconSecondary, this), "Export...");
    p_exportAction->setToolTip("Export the peak list");
    p_exportAction->setEnabled(false);
    connect(p_exportAction,&QAction::triggered,this,&PeakFindWidget::launchExportDialog);

    p_removeAction = p_toolBar2->addAction(ThemeColors::createThemedIcon(":/icons/minus.svg", ThemeColors::IconPrimary, this), "Remove");
    p_removeAction->setToolTip("Remove the selected peaks from the list");
    p_removeAction->setEnabled(false);
    connect(p_removeAction,&QAction::triggered,this,&PeakFindWidget::removeSelected);

    p_toolBar2->addSeparator();

    p_raiseParentAction = p_toolBar2->addAction(ThemeColors::createThemedIcon(":/icons/arrow-up.svg", ThemeColors::IconPrimary, this), "Show Parent");
    p_raiseParentAction->setToolTip("Bring the parent window to front");
    connect(p_raiseParentAction,&QAction::triggered,this,&PeakFindWidget::raiseParent);

    auto *spacer2 = new QWidget(this);
    spacer2->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    p_toolBar2->addWidget(spacer2);

    // Slim display-filter strip, hidden until the Filter action is on.
    p_filterStrip = new QWidget(this);
    auto *fl = new QHBoxLayout(p_filterStrip);
    fl->setContentsMargins(4,2,4,2);
    fl->setSpacing(4);

    p_minFreqBox = new QDoubleSpinBox(p_filterStrip);
    p_minFreqBox->setDecimals(3);
    p_minFreqBox->setSuffix(" MHz");
    p_minFreqBox->setKeyboardTracking(false);
    p_minFreqBox->setSpecialValueText("Min");
    p_minFreqBox->setToolTip("Hide peaks below this frequency. Display only — does not change the search range. At minimum: no lower bound.");

    p_maxFreqBox = new QDoubleSpinBox(p_filterStrip);
    p_maxFreqBox->setDecimals(3);
    p_maxFreqBox->setSuffix(" MHz");
    p_maxFreqBox->setKeyboardTracking(false);
    p_maxFreqBox->setSpecialValueText("Max");
    p_maxFreqBox->setToolTip("Hide peaks above this frequency. Display only — does not change the search range. At minimum: no upper bound.");

    p_minIntBox = new ScientificSpinBox(p_filterStrip);
    p_minIntBox->setToolTip("Hide peaks below this intensity. Display only. Zero: no intensity bound.");

    p_inViewBox = new QCheckBox("In view",p_filterStrip);
    p_inViewBox->setToolTip("Show only peaks within the main FT plot's currently visible frequency range.");

    fl->addWidget(new QLabel("Freq",p_filterStrip));
    fl->addWidget(p_minFreqBox);
    fl->addWidget(new QLabel(QString::fromUtf8("–"),p_filterStrip));
    fl->addWidget(p_maxFreqBox);
    fl->addSpacing(8);
    fl->addWidget(new QLabel("Min Int",p_filterStrip));
    fl->addWidget(p_minIntBox);
    fl->addStretch(1);
    fl->addWidget(p_inViewBox);
    p_filterStrip->setVisible(false);

    auto onFilterChanged = [this](){ applyFilters(); persistFilterState(); };
    connect(p_minFreqBox,qOverload<double>(&QDoubleSpinBox::valueChanged),this,onFilterChanged);
    connect(p_maxFreqBox,qOverload<double>(&QDoubleSpinBox::valueChanged),this,onFilterChanged);
    connect(p_minIntBox,&ScientificSpinBox::valueChanged,this,onFilterChanged);
    connect(p_inViewBox,&QCheckBox::toggled,this,onFilterChanged);

    p_peakListView = new QTableView(this);
    p_peakListView->setSelectionMode(QAbstractItemView::MultiSelection);
    p_peakListView->setSelectionBehavior(QAbstractItemView::SelectRows);
    p_peakListView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    p_peakListView->setAlternatingRowColors(true);
    p_peakListView->setSortingEnabled(true);
    p_peakListView->verticalHeader()->setVisible(false);

    // Both columns share the dock width so the table never needs a
    // horizontal scrollbar. Forcing the vertical scrollbar always-on keeps
    // the table's width constant when stacked docks fight for height —
    // otherwise its appearance/disappearance jolts the whole dock column.
    auto *hh = p_peakListView->horizontalHeader();
    hh->setStretchLastSection(false);
    hh->setSectionResizeMode(QHeaderView::Stretch);
    p_peakListView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    p_peakListView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    mainLayout->addWidget(p_toolBar);
    mainLayout->addWidget(p_toolBar2);
    mainLayout->addWidget(p_filterStrip);
    mainLayout->addWidget(p_peakListView,1);
}

void PeakFindWidget::applyFilters()
{
    constexpr double inf = std::numeric_limits<double>::infinity();

    // A spin box sitting at its minimum shows specialValueText and means
    // "no bound on this side".
    const double lo = (p_minFreqBox->value() <= p_minFreqBox->minimum())
                          ? -inf : p_minFreqBox->value();
    const double hi = (p_maxFreqBox->value() <= p_maxFreqBox->minimum())
                          ? inf : p_maxFreqBox->value();
    const double mi = (p_minIntBox->value() <= 0.0) ? -inf : p_minIntBox->value();
    const bool fen = p_filterAction->isChecked();

    p_proxy->setMinFreq(lo);
    p_proxy->setMaxFreq(hi);
    p_proxy->setMinIntensity(mi);
    p_proxy->setStaticFilterEnabled(fen);
    p_proxy->setViewSyncEnabled(fen && p_inViewBox->isChecked());
}

void PeakFindWidget::persistFilterState()
{
    set(BC::Key::pfFilterMinFreq,p_minFreqBox->value(),false);
    set(BC::Key::pfFilterMaxFreq,p_maxFreqBox->value(),false);
    set(BC::Key::pfFilterMinInt,p_minIntBox->value(),false);
    set(BC::Key::pfFilterEnabled,p_filterAction->isChecked(),false);
    set(BC::Key::pfViewSync,p_inViewBox->isChecked(),false);
    save();
}

void PeakFindWidget::setMainPlotXRange(double min, double max)
{
    // Fed unconditionally; the proxy only narrows the table while the
    // "In view" control is active.
    p_proxy->setViewRange(min,max);
}

void PeakFindWidget::newFt(const Ft ft)
{
    d_currentFt = ft;

    p_findAction->setEnabled(true);

    if(p_liveAction->isChecked())
        findPeaks();
}

void PeakFindWidget::newPeakList(const QVector<QPointF> pl)
{
    d_busy = false;

    //send peak list to model
    p_listModel->setPeakList(pl);
    emit peakList(p_listModel->peakList());

    p_exportAction->setEnabled(!pl.isEmpty());

    if(d_waiting)
        findPeaks();
}

void PeakFindWidget::findPeaks()
{
    if(d_currentFt.isEmpty())
        return;

    if(!d_busy)
    {
        d_busy = true;
        pu_watcher->setFuture(QtConcurrent::run([this](){p_pf->findPeaks(d_currentFt,d_minFreq,d_maxFreq,d_snr);}));
        d_waiting = false;
    }
    else
        d_waiting = true;
}

void PeakFindWidget::removeSelected()
{
    QModelIndexList l = p_peakListView->selectionModel()->selectedRows();
    if(!l.isEmpty())
    {
        QVector<int> rows;
        for(int i=0; i<l.size(); i++)
            rows.append(p_proxy->mapToSource(l.at(i)).row());

        p_listModel->removePeaks(rows);
    }

    emit peakList(p_listModel->peakList());
}

void PeakFindWidget::updateRemoveButton()
{
    p_removeAction->setEnabled(!p_peakListView->selectionModel()->selectedRows().isEmpty());
}

void PeakFindWidget::changeScaleFactor(double scf)
{
    if(p_listModel->rowCount(QModelIndex()) > 0)
    {
        p_listModel->scalingChanged(scf);
        emit peakList(p_listModel->peakList());
    }
}

void PeakFindWidget::launchOptionsDialog()
{
    QDialog d(this);
    d.setWindowTitle(QString("Peak Finding Options"));

    QFormLayout *fl = new QFormLayout;

    QDoubleSpinBox *minBox = new QDoubleSpinBox(&d);
    minBox->setDecimals(3);
    minBox->setRange(d_currentFt.minFreqMHz(),d_currentFt.maxFreqMHz());
    minBox->setValue(d_minFreq);
    minBox->setSuffix(QString(" MHz"));
    fl->addRow(QString("Min Frequency"),minBox);

    QDoubleSpinBox *maxBox = new QDoubleSpinBox(&d);
    maxBox->setDecimals(3);
    maxBox->setRange(d_currentFt.minFreqMHz(),d_currentFt.maxFreqMHz());
    maxBox->setValue(d_maxFreq);
    maxBox->setSuffix(QString(" MHz"));
    fl->addRow(QString("Max Frequency"),maxBox);

    QDoubleSpinBox *snrBox = new QDoubleSpinBox(&d);
    snrBox->setDecimals(1);
    snrBox->setRange(1.0,10000.0);
    snrBox->setValue(d_snr);
    snrBox->setToolTip(QString("Signal-to-noise ratio threshold for peak detection."));
    fl->addRow(QString("SNR"),snrBox);

    QSpinBox *winBox = new QSpinBox(&d);
    winBox->setRange(7,10001);
    winBox->setSingleStep(2);
    winBox->setValue(d_winSize);
    winBox->setToolTip(QString("Window size for Savitsky-Golay smoothing. Must be odd and greater than the polynomial order."));
    fl->addRow(QString("Window Size"),winBox);

    QSpinBox *orderBox = new QSpinBox(&d);
    orderBox->setRange(2,100);
    orderBox->setValue(d_polyOrder);
    orderBox->setToolTip(QString("Polynomial order for Savistsky-Golay smoothing. Must be less than the window size."));
    fl->addRow(QString("Polynomial Order"),orderBox);

    QVBoxLayout *vbl = new QVBoxLayout;
    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Ok|QDialogButtonBox::Cancel,&d);
    connect(bb->button(QDialogButtonBox::Ok),&QPushButton::clicked,&d,&QDialog::accept);
    connect(bb->button(QDialogButtonBox::Cancel),&QPushButton::clicked,&d,&QDialog::reject);

    vbl->addLayout(fl,1);
    vbl->addWidget(bb,0);

    d.setLayout(vbl);

    if(d.exec() == QDialog::Accepted)
    {
        double minFreq = minBox->value();
        double maxFreq = maxBox->value();
        int ws = winBox->value();
        int po = orderBox->value();

        if(ws < po+1)
            ws = po+1;

        if(!(ws % 2))
            ws++;

        if(minFreq > maxFreq)
            qSwap(minFreq,maxFreq);

        d_minFreq = minFreq;
        d_maxFreq = maxFreq;
        d_winSize = ws;
        d_polyOrder = po;
        d_snr = snrBox->value();

        QMetaObject::invokeMethod(p_pf,[this](){p_pf->calcCoefs(d_winSize,d_polyOrder);});

        set(BC::Key::pfMinFreq,d_minFreq,false);
        set(BC::Key::pfMaxFreq,d_maxFreq,false);
        set(BC::Key::pfSnr,d_snr,false);
        set(BC::Key::pfWinSize,d_winSize,false);
        set(BC::Key::pfOrder,d_polyOrder,false);
        save();

        if(p_liveAction->isChecked())
            findPeaks();
    }
}

void PeakFindWidget::launchExportDialog()
{
    PeakListExportDialog d(p_listModel->peakList(),d_number,this);
    d.exec();
}

void PeakFindWidget::raiseParent()
{
    // The immediate parent is the dock; raise the FtMW view's top-level
    // window (the floating dock itself when torn out).
    FtmwViewWidget *ftmwView = findFtmwView();
    QWidget *w = ftmwView ? ftmwView->window() : nullptr;
    if(w)
    {
        w->activateWindow();
        w->raise();
        w->show();
    }
}

FtmwViewWidget *PeakFindWidget::findFtmwView() const
{
    for (QWidget *w = parentWidget(); w != nullptr; w = w->parentWidget()) {
        if (auto *f = qobject_cast<FtmwViewWidget*>(w))
            return f;
    }
    return nullptr;
}

void PeakFindWidget::updateRaiseParentVisibility()
{
    if (!p_raiseParentAction)
        return;

    bool show = false;
    if (auto *dock = qobject_cast<QDockWidget*>(parentWidget()))
        show = dock->isFloating();
    else if (isWindow())
        show = parentWidget() != nullptr;
    p_raiseParentAction->setVisible(show);
}

void PeakFindWidget::adjustToolbarStyle()
{
    if (!p_toolBar || !p_toolBar2)
        return;

    // Measure both rows with labels shown; if either would overflow the
    // dock width, drop both to icon-only so the two rows stay visually
    // consistent (tooltips already describe each action).
    p_toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    p_toolBar2->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    const bool fits = p_toolBar->sizeHint().width() <= width()
                      && p_toolBar2->sizeHint().width() <= width();
    const auto style = fits ? Qt::ToolButtonTextBesideIcon
                            : Qt::ToolButtonIconOnly;
    p_toolBar->setToolButtonStyle(style);
    p_toolBar2->setToolButtonStyle(style);
}

void PeakFindWidget::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    adjustToolbarStyle();
}

void PeakFindWidget::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);

    // The hosting dock only becomes our parent after setWidget(), so the
    // float-state hookup can't happen in the constructor.
    if (!d_dockHooked) {
        if (auto *dock = qobject_cast<QDockWidget*>(parentWidget())) {
            connect(dock,&QDockWidget::topLevelChanged,
                    this,&PeakFindWidget::updateRaiseParentVisibility);
            d_dockHooked = true;
        }
    }
    updateRaiseParentVisibility();
    adjustToolbarStyle();
}
