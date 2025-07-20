#include "overlaymanagerwidget.h"
#include "unifiedoverlaydialog.h"
#include "overlayconfiguredelegate.h"
#include <gui/plot/curveappearancewidget.h>
#include <gui/plot/curveappearancepresetmanager.h>
#include <gui/plot/presetsavedialog.h>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QDebug>
#include <QTableView>
#include <QHeaderView>
#include <QMessageBox>
#include <QProgressBar>
#include <QCloseEvent>
#include <QMenu>
#include <QWidgetAction>
#include <QColorDialog>
#include <gui/widget/ftmwviewwidget.h>
#include <gui/plot/blackchirpplotcurve.h>

OverlayManagerWidget::OverlayManagerWidget(QWidget *parent, int number, const QVector<std::shared_ptr<OverlayBase>> &overlays)
    : QWidget{parent, Qt::Window}, SettingsStorage(BC::Key::OverlayManager::key), p_configureDelegate(nullptr), p_enabledDelegate(nullptr)
{
    // Set window attributes
    if(number > 0)
        setWindowTitle(QString("Overlay Manager: Experiment %1").arg(number));
    else
        setWindowTitle("Overlay Manager: Main Window");
    setWindowIcon(QIcon(":/icons/peak-find.svg")); // Temporary icon
    setAttribute(Qt::WA_DeleteOnClose);
    resize(900, 400); // Increased width to accommodate all columns including type

    setupUI();
    populateWithExistingOverlays(overlays);
    updateButtonStates();
    
    // Progress indicator starts hidden
    p_progressWidget->setVisible(false);
    
    // Restore window geometry if available
    QByteArray geom = get(BC::Key::OverlayManager::geometry).toByteArray();
    if (!geom.isEmpty()) {
        restoreGeometry(geom);
    }
}

OverlayManagerWidget::~OverlayManagerWidget()
{
    // Wait for any pending writes to complete before destruction
    if (p_overlayStorage) {
        p_overlayStorage->waitForPendingWrites();
        // Explicitly disconnect from overlay storage to prevent segfaults
        disconnect(p_overlayStorage.get(), nullptr, this, nullptr);
    }
}

void OverlayManagerWidget::setupUI()
{
    // Create main layout
    auto mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    // Create toolbar
    p_toolBar = new QToolBar(this);
    p_toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

    // Set up add button with dropdown
    setupAddButton();

    p_removeAction = p_toolBar->addAction(QIcon(":/icons/remove.png"), "Remove Overlay");
    p_removeAction->setToolTip("Remove the selected overlay");

    // Add separator
    p_toolBar->addSeparator();

    // Add raise parent action
    p_raiseParentAction = p_toolBar->addAction(QIcon(":/icons/go-up.svg"), "Show Parent");
    p_raiseParentAction->setToolTip("Bring the parent window to front");

    // Add spacer
    auto spacer = new QWidget;
    spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    p_toolBar->addWidget(spacer);

    // Create unified table view
    p_overlayModel = new OverlayTableModel(this);
    p_overlayTableView = new QTableView(this);
    p_overlayTableView->setModel(p_overlayModel);
    
    // Connect model signals to track overlay changes
    connect(p_overlayModel, &OverlayTableModel::dataChanged, 
            this, &OverlayManagerWidget::onModelDataChanged);
    connect(p_overlayModel, &OverlayTableModel::rowsInserted, 
            this, [this]() { resizeColumnsToContents(); });
    connect(p_overlayModel, &OverlayTableModel::rowsRemoved, 
            this, [this]() { resizeColumnsToContents(); });
    
    // Connect selection signals to update button states
    connect(p_overlayTableView->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &OverlayManagerWidget::onSelectionChanged);

    // Configure table view
    p_overlayTableView->setSelectionBehavior(QAbstractItemView::SelectRows);
    p_overlayTableView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    p_overlayTableView->setAlternatingRowColors(true);
    p_overlayTableView->setSortingEnabled(true); // Enable column header sorting
    
    // Enable drag and drop for reordering
    p_overlayTableView->setDragDropMode(QAbstractItemView::InternalMove);
    p_overlayTableView->setDefaultDropAction(Qt::MoveAction);
    p_overlayTableView->setDragEnabled(true);
    p_overlayTableView->setAcceptDrops(true);
    p_overlayTableView->setDropIndicatorShown(true);
    
    // Enable context menu
    p_overlayTableView->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(p_overlayTableView, &QTableView::customContextMenuRequested,
            this, &OverlayManagerWidget::showContextMenu);
    
    // Enable double-click to configure overlay
    connect(p_overlayTableView, &QTableView::doubleClicked,
            this, &OverlayManagerWidget::onConfigureClicked);

    // Configure headers
    auto horizontalHeader = p_overlayTableView->horizontalHeader();
    horizontalHeader->setStretchLastSection(false);
    horizontalHeader->setSectionResizeMode(QHeaderView::Interactive);

    auto verticalHeader = p_overlayTableView->verticalHeader();
    verticalHeader->setDefaultSectionSize(25);
    verticalHeader->setVisible(false);

    // Set up table view with delegates and column widths
    setupTableView();

    // Create progress indicator widget
    createProgressWidget();
    
    // Add widgets to layout
    mainLayout->addWidget(p_toolBar);
    mainLayout->addWidget(p_overlayTableView);
    mainLayout->addWidget(p_progressWidget);

    // Connect signals
    connect(p_removeAction, &QAction::triggered, this, &OverlayManagerWidget::removeOverlay);
    connect(p_raiseParentAction, &QAction::triggered, this, &OverlayManagerWidget::raiseParent);
    
    // Set up keyboard shortcuts
    setupKeyboardShortcuts();
}

void OverlayManagerWidget::setupAddButton()
{
    // Create add button with dropdown menu
    p_addButton = new QToolButton(this);
    p_addButton->setIcon(QIcon(":/icons/add.png"));
    p_addButton->setText("Add Overlay");
    p_addButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    p_addButton->setPopupMode(QToolButton::InstantPopup);
    p_addButton->setToolTip("Add a new overlay to the current plot");

    // Create dropdown menu
    p_addMenu = new QMenu(this);
    
    // Use Q_ENUM to get all overlay types dynamically
    auto metaEnum = QMetaEnum::fromType<OverlayBase::OverlayType>();

    for(int i = 0; i < metaEnum.keyCount(); ++i)
    {
        QString typeName = metaEnum.key(i);
        auto typeValue = static_cast<OverlayBase::OverlayType>(metaEnum.value(i));

        // Create friendly names for menu items
        QString menuItemName;
        bool enabled = true;

        switch(typeValue)
        {
        case OverlayBase::BCExperiment:
            menuItemName = "BC Experiment";
            break;
        case OverlayBase::Catalog:
            menuItemName = "Catalog";
            enabled = true; // Now implemented
            break;
        case OverlayBase::GenericXY:
            menuItemName = "Generic XY Data";
            enabled = true; // Now implemented
            break;
        default:
            menuItemName = typeName; // Fallback to enum name
            enabled = false;
            break;
        }

        // Create action for this overlay type
        QAction *action = p_addMenu->addAction(QIcon(":/icons/add.png"), menuItemName);
        action->setEnabled(enabled);
        d_addActions[typeValue] = action;

        // Connect action to addOverlay slot with the specific type
        connect(action, &QAction::triggered, this, [this, typeValue]() {
            addOverlay(typeValue);
        });
    }

    p_addButton->setMenu(p_addMenu);
    p_toolBar->addWidget(p_addButton);
}


void OverlayManagerWidget::updateButtonStates()
{
    // Check if we have pending writes
    bool hasPendingWrites = p_overlayStorage && p_overlayStorage->hasPendingWrites();
    
    // Add button is disabled when there are pending writes
    p_addButton->setEnabled(!hasPendingWrites);
    
    // Remove button is enabled only when rows are selected and no pending writes
    bool hasSelection = false;
    
    if (p_overlayTableView && p_overlayTableView->selectionModel()) {
        hasSelection = p_overlayTableView->selectionModel()->hasSelection();
    }
    
    p_removeAction->setEnabled(hasSelection && !hasPendingWrites);
}

void OverlayManagerWidget::addOverlay(OverlayBase::OverlayType type)
{
    // Ensure we have overlay storage connection
    if (!p_overlayStorage) {
        qDebug() << "Warning: No overlay storage connected to OverlayManagerWidget";
        return;
    }

    // Get the FtmwViewWidget parent for dialog constructors
    FtmwViewWidget* ftmwParent = qobject_cast<FtmwViewWidget*>(parentWidget());
    if(!ftmwParent) {
        return;
    }
    
    // All overlay types are now implemented
    
    // Get plot information from parent
    Ft mainFt = ftmwParent->getMainPlotFt();
    QStringList plotNames = ftmwParent->getPlotNames();
    
    // Get existing overlays for context
    QVector<std::shared_ptr<OverlayBase>> existingOverlays = p_overlayModel->getAllOverlays();
    
    // Create unified dialog in creation mode with full Ft data for intelligent settings
    UnifiedOverlayDialog dialog(type, plotNames, mainFt, existingOverlays, this);
    dialog.setModal(true);
    
    // Connect preview signals for real-time preview display
    connect(&dialog, &UnifiedOverlayDialog::previewRequested,
            this, &OverlayManagerWidget::onPreviewRequested);
    connect(&dialog, &UnifiedOverlayDialog::previewCancelled,
            this, &OverlayManagerWidget::onPreviewCancelled);
    connect(&dialog, &UnifiedOverlayDialog::previewOverlayRequested,
            this, &OverlayManagerWidget::onPreviewOverlayRequested);
    connect(&dialog, &UnifiedOverlayDialog::previewOverlayCancelled,
            this, &OverlayManagerWidget::onPreviewOverlayCancelled);
    connect(&dialog, &UnifiedOverlayDialog::overlayDataChanged,
            this, &OverlayManagerWidget::onOverlaySettingsChanged);
    
    // Run the dialog and get the overlay if accepted
    if(dialog.exec() == QDialog::Accepted) {
        auto overlay = dialog.getOverlay();
        
        // Add overlay to storage if created successfully
        if (overlay != nullptr) {
            // Check if this overlay was previously a preview overlay and remove it from preview storage
            // This ensures proper transfer from preview to regular storage
            if (p_overlayStorage) {
                auto previewOverlays = p_overlayStorage->getAllPreviewOverlays();
                for (const auto &previewOverlay : previewOverlays) {
                    if (previewOverlay.get() == overlay.get()) {
                        // This overlay came from preview storage - remove it from there first
                        p_overlayStorage->removePreviewOverlay(overlay->getLabel());
                        break;
                    }
                }
            }
            
            // Ensure overlay is not in preview mode for persistent storage
            overlay->setPreview(false);
            overlay->setEnabled(true);
            
            // Add directly to overlay storage - this initiates async write
            if (p_overlayStorage->addOverlay(overlay)) {
                // Add to unified model for display
                p_overlayModel->addOverlay(overlay);
                
                // Update UI state to show any pending writes
                updateButtonStates();
            }
        }
    }
    
    // Clear any remaining preview overlays after dialog closes
    if (p_overlayStorage) {
        p_overlayStorage->clearAllPreviews();
    }
}

void OverlayManagerWidget::removeOverlay()
{
    if (!p_overlayTableView || !p_overlayModel)
        return;
        
    auto selectionModel = p_overlayTableView->selectionModel();
    if (!selectionModel->hasSelection())
        return;
        
    // Get selected rows
    QModelIndexList selectedRows = selectionModel->selectedRows();
    if (selectedRows.isEmpty())
        return;
        
    // Create confirmation message
    QString message;
    if (selectedRows.size() == 1) {
        auto overlay = p_overlayModel->getOverlay(selectedRows.first().row());
        if (overlay) {
            message = QString("Are you sure you want to remove the overlay \"%1\"?").arg(overlay->getLabel());
        } else {
            message = "Are you sure you want to remove the selected overlay?";
        }
    } else {
        message = QString("Are you sure you want to remove %1 selected overlays?").arg(selectedRows.size());
    }
    
    // Show confirmation dialog
    int result = QMessageBox::question(this, "Remove Overlay", message,
                                     QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    
    if (result != QMessageBox::Yes)
        return;
        
    // Ensure we have overlay storage connection
    if (!p_overlayStorage) {
        qDebug() << "Warning: No overlay storage connected to OverlayManagerWidget";
        return;
    }
    
    // Remove overlays in reverse order to maintain valid indices
    QVector<int> rows;
    for (const auto& index : selectedRows) {
        rows.append(index.row());
    }
    std::sort(rows.begin(), rows.end(), std::greater<int>());
    
    for (int row : rows) {
        auto overlay = p_overlayModel->getOverlay(row);
        if (overlay) {
            // Remove from overlay storage - this will emit signals that FtmwViewWidget listens to
            if (p_overlayStorage->removeOverlay(overlay->getLabel())) {
                // Remove from unified model for display
                p_overlayModel->removeOverlay(row);
                
            } else {
                qDebug() << "Failed to remove overlay from storage";
            }
        }
    }
}

void OverlayManagerWidget::raiseParent()
{
    // Get parent widget and bring it to front
    QWidget* w = parentWidget();
    if(w)
    {
        w->activateWindow();
        w->raise();
        w->show();
    }
}

void OverlayManagerWidget::populateWithExistingOverlays(const QVector<std::shared_ptr<OverlayBase>> &overlays)
{
    // Add existing overlays to the unified model
    for(const auto& overlay : overlays)
    {
        if(overlay == nullptr)
            continue;
            
        // All overlay types are now handled by the unified model
        if(p_overlayModel != nullptr)
            p_overlayModel->addOverlay(overlay);
    }
}

void OverlayManagerWidget::setupConfigureDelegate()
{
    // Create and set the delegate for the Configure column
    p_configureDelegate = new OverlayConfigureDelegate(this);
    p_overlayTableView->setItemDelegateForColumn(static_cast<int>(OverlayTableModel::ConfigureColumn), p_configureDelegate);
    
    // Connect the delegate signal to handle configuration button clicks
    connect(p_configureDelegate, &OverlayConfigureDelegate::configureClicked,
            this, &OverlayManagerWidget::onConfigureClicked);
}

void OverlayManagerWidget::setupEnabledDelegate()
{
    // Create and set the delegate for the Enabled column
    p_enabledDelegate = new OverlayCheckBoxDelegate(this);
    p_overlayTableView->setItemDelegateForColumn(static_cast<int>(OverlayTableModel::EnabledColumn), p_enabledDelegate);
}

void OverlayManagerWidget::setupTableView()
{
    // Set up delegates
    setupConfigureDelegate();
    setupEnabledDelegate();
    
    // Set up column resize behavior
    resizeColumnsToContents();
}

void OverlayManagerWidget::setupKeyboardShortcuts()
{
    // Ctrl+C: Copy appearance 
    p_copyAppearanceShortcut = new QShortcut(QKeySequence::Copy, this);
    connect(p_copyAppearanceShortcut, &QShortcut::activated, this, [this]() {
        auto selectionInfo = getSelectionInfo();
        if (selectionInfo.singleRowSelected) {
            copyAppearanceSettings(selectionInfo.overlay);
        } else {
            // Clear clipboard and provide feedback for invalid selection
            /// TODO: Display message on UI
            d_clipboardAppearance.clear();
            if (selectionInfo.multipleRowsSelected) {
                qDebug() << "Cannot copy appearance: multiple rows selected. Clipboard cleared.";
            } else {
                qDebug() << "Cannot copy appearance: no row selected. Clipboard cleared.";
            }
        }
    });
    
    // Ctrl+V: Paste appearance
    p_pasteAppearanceShortcut = new QShortcut(QKeySequence::Paste, this);
    connect(p_pasteAppearanceShortcut, &QShortcut::activated, this, [this]() {
        if (hasClipboardAppearance()) {
            pasteAppearanceToSelected();
        }
    });
    
    // Ctrl+Shift+C: Copy overlay settings
    p_copySettingsShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_C), this);
    connect(p_copySettingsShortcut, &QShortcut::activated, this, [this]() {
        auto selectionInfo = getSelectionInfo();
        if (selectionInfo.singleRowSelected) {
            copyOverlaySettings(selectionInfo.overlay);
        } else {
            // Clear clipboard and provide feedback for invalid selection
            /// TODO: Display message on UI
            d_clipboardSettings.clear();
            if (selectionInfo.multipleRowsSelected) {
                qDebug() << "Cannot copy overlay settings: multiple rows selected. Clipboard cleared.";
            } else {
                qDebug() << "Cannot copy overlay settings: no row selected. Clipboard cleared.";
            }
        }
    });
    
    // Ctrl+Shift+V: Paste overlay settings  
    p_pasteSettingsShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_V), this);
    connect(p_pasteSettingsShortcut, &QShortcut::activated, this, [this]() {
        if (hasClipboardSettings()) {
            pasteSettingsToSelected();
        }
    });
    
    // Ctrl+Z: Undo last paste operation
    p_undoShortcut = new QShortcut(QKeySequence::Undo, this);
    connect(p_undoShortcut, &QShortcut::activated, this, [this]() {
        if (d_undoData.hasUndoData) {
            performUndo();
        }
    });
}

OverlayManagerWidget::SelectionInfo OverlayManagerWidget::getSelectionInfo()
{
    SelectionInfo info;
    
    if (!p_overlayTableView || !p_overlayModel) {
        return info;
    }
    
    auto selectionModel = p_overlayTableView->selectionModel();
    if (!selectionModel->hasSelection()) {
        return info;
    }
    
    // Get selected rows and analyze selection
    QModelIndexList selectedRows = selectionModel->selectedRows();
    info.selectedCount = selectedRows.size();
    
    if (info.selectedCount == 1) {
        info.singleRowSelected = true;
        info.overlay = p_overlayModel->getOverlay(selectedRows.first().row());
    } else if (info.selectedCount > 1) {
        info.multipleRowsSelected = true;
    }
    
    return info;
}

std::shared_ptr<OverlayBase> OverlayManagerWidget::getSelectedOverlay()
{
    // Backward compatibility method - only return overlay if exactly one row is selected
    auto info = getSelectionInfo();
    return info.singleRowSelected ? info.overlay : nullptr;
}

void OverlayManagerWidget::captureUndoState(const QVector<std::shared_ptr<OverlayBase>> &overlays, const QString &operationType)
{
    if (overlays.isEmpty()) {
        return;
    }
    
    // Clear any existing undo data
    d_undoData = UndoData();
    
    // Store basic undo information
    d_undoData.hasUndoData = true;
    d_undoData.operationType = operationType;
    d_undoData.overlayCount = overlays.size();
    
    // Capture current state for all overlays
    for (auto overlay : overlays) {
        if (!overlay) {
            continue;
        }
        
        OverlayUndoState undoState;
        undoState.overlay = overlay;
        
        // Capture current state based on operation type
        if (operationType == "appearance" || operationType == "both") {
            // Store current appearance metadata
            undoState.previousAppearanceData["curveColor"] = overlay->getCurveMetadata(BC::Key::bcCurveColor);
            undoState.previousAppearanceData["curveCurveStyle"] = overlay->getCurveMetadata(BC::Key::bcCurveCurveStyle);
            undoState.previousAppearanceData["curveThickness"] = overlay->getCurveMetadata(BC::Key::bcCurveThickness);
            undoState.previousAppearanceData["curveLineStyle"] = overlay->getCurveMetadata(BC::Key::bcCurveLineStyle);
            undoState.previousAppearanceData["curveMarker"] = overlay->getCurveMetadata(BC::Key::bcCurveMarker);
            undoState.previousAppearanceData["curveMarkerSize"] = overlay->getCurveMetadata(BC::Key::bcCurveMarkerSize);
            undoState.previousAppearanceData["curveVisible"] = overlay->getCurveMetadata(BC::Key::bcCurveVisible);
            undoState.previousAppearanceData["curveAutoscale"] = overlay->getCurveMetadata(BC::Key::bcCurveAutoscale);
            undoState.previousAppearanceData["curveAxisY"] = overlay->getCurveMetadata(BC::Key::bcCurveAxisY);
        }
        
        if (operationType == "settings" || operationType == "both") {
            // Store current overlay settings
            undoState.previousSettingsData["plotId"] = overlay->getPlotId();
            undoState.previousSettingsData["yScale"] = overlay->getYScale();
            undoState.previousSettingsData["yOffset"] = overlay->getYOffset();
            undoState.previousSettingsData["xOffset"] = overlay->getXOffset();
            undoState.previousSettingsData["minFreqEnabled"] = overlay->getMinFreqEnabled();
            undoState.previousSettingsData["minFreqValue"] = overlay->getMinFreqValue();
            undoState.previousSettingsData["maxFreqEnabled"] = overlay->getMaxFreqEnabled();
            undoState.previousSettingsData["maxFreqValue"] = overlay->getMaxFreqValue();
            undoState.previousSettingsData["enabled"] = overlay->getEnabled();
        }
        
        d_undoData.overlayStates.append(undoState);
    }
    
}

void OverlayManagerWidget::performUndo()
{
    if (!d_undoData.hasUndoData || d_undoData.overlayStates.isEmpty()) {
        /// TODO: Display message on UI
        qDebug() << "No undo data available";
        return;
    }
    
    // Restore previous state for all overlays
    for (const auto& undoState : d_undoData.overlayStates) {
        if (!undoState.overlay) {
            continue;
        }
        
        // Restore appearance data if needed
        if (d_undoData.operationType == "appearance" || d_undoData.operationType == "both") {
            for (auto it = undoState.previousAppearanceData.constBegin(); 
                 it != undoState.previousAppearanceData.constEnd(); ++it) {
                
                if (it.key() == "curveColor") {
                    undoState.overlay->setCurveMetadata(BC::Key::bcCurveColor, it.value());
                } else if (it.key() == "curveCurveStyle") {
                    undoState.overlay->setCurveMetadata(BC::Key::bcCurveCurveStyle, it.value());
                } else if (it.key() == "curveThickness") {
                    undoState.overlay->setCurveMetadata(BC::Key::bcCurveThickness, it.value());
                } else if (it.key() == "curveLineStyle") {
                    undoState.overlay->setCurveMetadata(BC::Key::bcCurveLineStyle, it.value());
                } else if (it.key() == "curveMarker") {
                    undoState.overlay->setCurveMetadata(BC::Key::bcCurveMarker, it.value());
                } else if (it.key() == "curveMarkerSize") {
                    undoState.overlay->setCurveMetadata(BC::Key::bcCurveMarkerSize, it.value());
                } else if (it.key() == "curveVisible") {
                    undoState.overlay->setCurveMetadata(BC::Key::bcCurveVisible, it.value());
                } else if (it.key() == "curveAutoscale") {
                    undoState.overlay->setCurveMetadata(BC::Key::bcCurveAutoscale, it.value());
                } else if (it.key() == "curveAxisY") {
                    undoState.overlay->setCurveMetadata(BC::Key::bcCurveAxisY, it.value());
                }
            }
        }
        
        // Restore settings data if needed
        if (d_undoData.operationType == "settings" || d_undoData.operationType == "both") {
            if (undoState.previousSettingsData.contains("plotId")) {
                undoState.overlay->setPlotId(undoState.previousSettingsData["plotId"].toString());
            }
            if (undoState.previousSettingsData.contains("yScale")) {
                undoState.overlay->setYScale(undoState.previousSettingsData["yScale"].toDouble());
            }
            if (undoState.previousSettingsData.contains("yOffset")) {
                undoState.overlay->setYOffset(undoState.previousSettingsData["yOffset"].toDouble());
            }
            if (undoState.previousSettingsData.contains("xOffset")) {
                undoState.overlay->setXOffset(undoState.previousSettingsData["xOffset"].toDouble());
            }
            if (undoState.previousSettingsData.contains("minFreqEnabled") && 
                undoState.previousSettingsData.contains("minFreqValue")) {
                undoState.overlay->setMinFreqLimit(undoState.previousSettingsData["minFreqEnabled"].toBool(),
                                                   undoState.previousSettingsData["minFreqValue"].toDouble());
            }
            if (undoState.previousSettingsData.contains("maxFreqEnabled") && 
                undoState.previousSettingsData.contains("maxFreqValue")) {
                undoState.overlay->setMaxFreqLimit(undoState.previousSettingsData["maxFreqEnabled"].toBool(),
                                                   undoState.previousSettingsData["maxFreqValue"].toDouble());
            }
            if (undoState.previousSettingsData.contains("enabled")) {
                undoState.overlay->setEnabled(undoState.previousSettingsData["enabled"].toBool());
            }
        }
        
        // Emit signal to update this overlay's display
        emit overlayDataChanged(undoState.overlay);
    }

    /// TODO: Display message on UI
    qDebug() << "Undid" << d_undoData.operationType << "operation on" << d_undoData.overlayCount << "overlays";
    
    // Clear undo data after use (only one level of undo)
    invalidateUndo();
}

void OverlayManagerWidget::invalidateUndo()
{
    d_undoData = UndoData();
}

QString OverlayManagerWidget::getUndoDescription() const
{
    if (!d_undoData.hasUndoData || d_undoData.overlayCount == 0) {
        return QString();
    }
    
    QString operationName;
    if (d_undoData.operationType == "appearance") {
        operationName = "Appearance Paste";
    } else if (d_undoData.operationType == "settings") {
        operationName = "Settings Paste";
    } else {
        operationName = "Paste";
    }
    
    if (d_undoData.overlayCount == 1) {
        return QString("Undo %1").arg(operationName);
    } else {
        return QString("Undo %1 to %2 overlays").arg(operationName).arg(d_undoData.overlayCount);
    }
}

void OverlayManagerWidget::resizeColumnsToContents()
{
    if (!p_overlayModel || !p_overlayTableView) {
        return;
    }
    
    auto horizontalHeader = p_overlayTableView->horizontalHeader();
    int columnCount = p_overlayModel->columnCount();
    int sourceFileColumn = static_cast<int>(OverlayTableModel::SourceFileColumn);
    
    // Resize all columns except the source file column to contents
    for (int i = 0; i < columnCount; ++i) {
        if (i != sourceFileColumn) {
            p_overlayTableView->resizeColumnToContents(i);
            // Configure and Enabled columns should be fixed width, others interactive
            if (i == static_cast<int>(OverlayTableModel::ConfigureColumn) || 
                i == static_cast<int>(OverlayTableModel::EnabledColumn)) {
                horizontalHeader->setSectionResizeMode(i, QHeaderView::Fixed);
            } else {
                horizontalHeader->setSectionResizeMode(i, QHeaderView::Interactive);
            }
        }
    }
    
    // Set the source file column to stretch to fill remaining space
    if (sourceFileColumn < columnCount) {
        horizontalHeader->setSectionResizeMode(sourceFileColumn, QHeaderView::Stretch);
    }
    
    // Set fixed width for configure and enabled columns with minimal padding
    QFontMetrics fm(p_overlayTableView->font());
    int configureWidth = fm.horizontalAdvance("⚙") + 8; // Gear symbol plus minimal padding
    int enabledWidth = fm.horizontalAdvance("👁") + 8; // Eye symbol plus minimal padding
    
    if (static_cast<int>(OverlayTableModel::ConfigureColumn) < columnCount) {
        p_overlayTableView->setColumnWidth(static_cast<int>(OverlayTableModel::ConfigureColumn), configureWidth);
    }
    if (static_cast<int>(OverlayTableModel::EnabledColumn) < columnCount) {
        p_overlayTableView->setColumnWidth(static_cast<int>(OverlayTableModel::EnabledColumn), enabledWidth);
    }
}

void OverlayManagerWidget::onModelDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight)
{
    // Invalidate undo when table data is changed through direct editing
    invalidateUndo();
    
    // Get the model that emitted the signal
    const OverlayTableModel* model = qobject_cast<const OverlayTableModel*>(topLeft.model());
    if (!model) {
        return;
    }
    
    // Process each changed cell
    for (int row = topLeft.row(); row <= bottomRight.row(); ++row) {
        for (int col = topLeft.column(); col <= bottomRight.column(); ++col) {
            auto overlay = model->getOverlay(row);
            if (!overlay) {
                continue;
            }
            
            // Check which column was changed and emit appropriate signal
            switch (col) {
            case OverlayTableModel::ConfigureColumn: // Configure - handled by delegate
            case OverlayTableModel::LabelColumn: // Label - doesn't affect plot display
            case OverlayTableModel::PlotIdColumn: // PlotId - not directly editable anymore
            case OverlayTableModel::SourceFileColumn: // SourceFile - doesn't affect plot display
            case OverlayTableModel::OverlayTypeColumn: // Type - not editable
                // No signal needed for these columns
                break;
            case OverlayTableModel::EnabledColumn: // Enabled - affects plot visibility
                // Emit signal to update plot visibility
                emit overlayDataChanged(overlay);
                break;
            default:
                // This should not happen for other columns
                break;
            }
        }
    }
    
    // Resize columns to contents for the unified view
    resizeColumnsToContents();
}

void OverlayManagerWidget::onSelectionChanged()
{
    // Update button states when selection changes
    updateButtonStates();
}

void OverlayManagerWidget::onConfigureClicked(const QModelIndex &index)
{
    // Get the overlay for this row
    auto overlay = p_overlayModel->getOverlay(index.row());
    if (!overlay) {
        return;
    }
    
    // Get the FtmwViewWidget parent to access plot names and xRange
    FtmwViewWidget* ftmwParent = qobject_cast<FtmwViewWidget*>(parentWidget());
    if (!ftmwParent) {
        QMessageBox::warning(this, "Error", "Cannot access parent widget for configuration.");
        return;
    }
    
    // Get the main plot Ft and plot names
    Ft mainFt = ftmwParent->getMainPlotFt();
    QStringList plotNames = ftmwParent->getPlotNames();
    
    // Invalidate undo when opening configure dialog
    invalidateUndo();
    
    // Create unified dialog in settings mode with full Ft data for intelligent settings
    UnifiedOverlayDialog dialog(overlay, plotNames, mainFt, p_overlayStorage, this);
    dialog.setModal(true);
    
    // Connect the dialog signal to our slot for real-time updates
    connect(&dialog, &UnifiedOverlayDialog::overlayDataChanged,
            this, &OverlayManagerWidget::onOverlaySettingsChanged);
    
    dialog.exec();
}

void OverlayManagerWidget::onOverlaySettingsChanged(std::shared_ptr<OverlayBase> overlay)
{
    // Emit signal for real-time plot updates
    emit overlayDataChanged(overlay);
    
    // Update the table model to reflect any changes
    // Find the row for this overlay and emit dataChanged for the entire row
    auto overlays = p_overlayModel->getAllOverlays();
    for (int i = 0; i < overlays.size(); ++i) {
        if (overlays[i] == overlay) {
            auto topLeft = p_overlayModel->index(i, 0);
            auto bottomRight = p_overlayModel->index(i, p_overlayModel->columnCount() - 1);
            emit p_overlayModel->dataChanged(topLeft, bottomRight);
            break;
        }
    }
}

void OverlayManagerWidget::createProgressWidget()
{
    // Create progress widget container
    p_progressWidget = new QWidget(this);
    auto progressLayout = new QHBoxLayout(p_progressWidget);
    progressLayout->setContentsMargins(5, 5, 5, 5);
    progressLayout->setSpacing(10);
    
    // Create progress label
    p_progressLabel = new QLabel("Writing overlay data...", p_progressWidget);
    p_progressLabel->setStyleSheet("font-weight: bold; color: #0066CC;");
    
    // Create progress bar
    p_progressBar = new QProgressBar(p_progressWidget);
    p_progressBar->setRange(0, 0); // Indeterminate progress
    p_progressBar->setMaximumHeight(16);
    
    // Add widgets to layout
    progressLayout->addWidget(p_progressLabel);
    progressLayout->addWidget(p_progressBar);
    progressLayout->addStretch();
    
    // Set widget background
    p_progressWidget->setStyleSheet("background-color: #F0F8FF; border: 1px solid #CCCCCC; border-radius: 4px;");
}

void OverlayManagerWidget::updateProgressDisplay(int pendingCount)
{
    if (pendingCount > 0) {
        // Show progress indicator
        if (pendingCount == 1) {
            p_progressLabel->setText("Writing 1 overlay...");
        } else {
            p_progressLabel->setText(QString("Writing %1 overlays...").arg(pendingCount));
        }
        p_progressWidget->setVisible(true);
    } else {
        // Hide progress indicator
        p_progressWidget->setVisible(false);
    }
}

void OverlayManagerWidget::showErrorNotification(const QString& overlayLabel, const QString& error)
{
    QString title = "Overlay Write Failed";
    QString message = QString("Failed to write overlay data for '%1'\n\nError: %2\n\nThe overlay has been removed.")
                     .arg(overlayLabel, error);
    
    QMessageBox::warning(this, title, message);
}

void OverlayManagerWidget::connectToOverlayStorage(std::shared_ptr<OverlayStorage> storage)
{
    // Disconnect from previous storage if any
    if (p_overlayStorage) {
        disconnect(p_overlayStorage.get(), nullptr, this, nullptr);
    }
    
    // Store new storage reference
    p_overlayStorage = storage;
    
    // Connect to storage signals
    if (p_overlayStorage) {
        connect(p_overlayStorage.get(), &OverlayStorage::overlayWriteCompleted,
                this, &OverlayManagerWidget::onOverlayWriteCompleted);
        connect(p_overlayStorage.get(), &OverlayStorage::overlayWriteFailed,
                this, &OverlayManagerWidget::onOverlayWriteFailed);
        connect(p_overlayStorage.get(), &OverlayStorage::pendingWritesChanged,
                this, &OverlayManagerWidget::onPendingWritesChanged);
        
        // Initialize progress display with current pending count
        updateProgressDisplay(p_overlayStorage->pendingWriteCount());
    }
}

void OverlayManagerWidget::onOverlayWriteCompleted(std::shared_ptr<OverlayBase> overlay)
{
    Q_UNUSED(overlay)
    // Write completed successfully - no action needed
    // Progress display will be updated via onPendingWritesChanged
}

void OverlayManagerWidget::onOverlayWriteFailed(std::shared_ptr<OverlayBase> overlay, QString error)
{
    if (!overlay) {
        return;
    }
    
    // Show error notification
    showErrorNotification(overlay->getLabel(), error);
    
    // Find and remove the overlay from the unified model
    if (p_overlayModel) {
        auto overlays = p_overlayModel->getAllOverlays();
        for (int i = 0; i < overlays.size(); ++i) {
            if (overlays[i] == overlay) {
                p_overlayModel->removeOverlay(i);
                break;
            }
        }
    }
}

void OverlayManagerWidget::onPendingWritesChanged(int count)
{
    // Update progress display
    updateProgressDisplay(count);
    
    // Update button states to potentially disable/enable overlay creation
    updateButtonStates();
}

void OverlayManagerWidget::onExternalOverlayDataChanged(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay || !p_overlayModel) {
        return;
    }
    
    // Find the overlay in the model and refresh its data
    auto overlays = p_overlayModel->getAllOverlays();
    for (int i = 0; i < overlays.size(); ++i) {
        if (overlays[i] == overlay) {
            // Emit dataChanged for the entire row to refresh all columns
            auto topLeft = p_overlayModel->index(i, 0);
            auto bottomRight = p_overlayModel->index(i, p_overlayModel->columnCount() - 1);
            emit p_overlayModel->dataChanged(topLeft, bottomRight);
            break;
        }
    }
}

void OverlayManagerWidget::showContextMenu(const QPoint &position)
{
    // Get the item at the clicked position
    QModelIndex index = p_overlayTableView->indexAt(position);
    if (!index.isValid()) {
        return; // No item clicked
    }
    
    // Get the overlay for this row
    auto overlay = p_overlayModel->getOverlay(index.row());
    if (!overlay) {
        return;
    }
    
    // Create context menu
    QMenu contextMenu(this);
    
    // === OVERLAY CONFIGURATION GROUP ===
    // Add Configure action with settings icon
    QAction *configureAction = contextMenu.addAction(QIcon(":/icons/configure.svg"), "Configure...");
    configureAction->setToolTip("Open overlay configuration dialog");
    connect(configureAction, &QAction::triggered, [this, index]() {
        onConfigureClicked(index);
    });
    
    // === CURVE APPEARANCE GROUP ===
    contextMenu.addSeparator();
    
    // Create Curve Appearance submenu
    QMenu *appearanceMenu = contextMenu.addMenu(QIcon(":/icons/palette.svg"), "Curve Appearance");
    
    // Add Curve Appearance widget to submenu
    auto curveWa = new QWidgetAction(appearanceMenu);
    auto appearanceWidget = new CurveAppearanceWidget(appearanceMenu);
    
    // Connect to global preset manager for preset functionality
    appearanceWidget->setPresetManager(CurveAppearancePresetManager::instance());
    
    // Initialize widget from overlay metadata
    appearanceWidget->initializeFromOverlay(overlay);
    
    // Connect widget to real-time overlay metadata updates
    connect(appearanceWidget, &CurveAppearanceWidget::curveAppearanceChanged,
            this, [this, overlay, appearanceWidget](const CurveAppearanceWidget::CurveAppearance &) {
        // Save appearance to overlay metadata
        appearanceWidget->applyToOverlay(overlay);
        
        // Emit signal for real-time plot updates
        emit overlayDataChanged(overlay);
    });
    
    // Handle color change requests with color dialog
    connect(appearanceWidget, &CurveAppearanceWidget::colorChangeRequested,
            this, [this, overlay, appearanceWidget]() {
        // Get current color from overlay metadata
        QColor currentColor = overlay->getCurveMetadata(BC::Key::bcCurveColor).value<QColor>();
        if (!currentColor.isValid()) {
            currentColor = palette().color(QPalette::Text);
        }
        
        // Open color dialog
        QColor newColor = QColorDialog::getColor(currentColor, this, "Choose Curve Color");
        
        // Update widget and overlay if valid color chosen
        if (newColor.isValid()) {
            appearanceWidget->updateColorDisplay(newColor);
        }
    });
    
    // Handle preset save requests with custom dialog
    connect(appearanceWidget, &CurveAppearanceWidget::presetSaveRequested,
            this, [this, appearanceWidget](const QString &suggestedName) {
        PresetSaveDialog dialog(suggestedName, CurveAppearancePresetManager::instance(), this);
        
        if (dialog.exec() == QDialog::Accepted) {
            QString presetName = dialog.getPresetName();
            if (!presetName.isEmpty()) {
                // If overwriting existing preset, no additional confirmation needed
                // since the dialog already handled the selection
                if (!dialog.isOverwriteMode()) {
                    // For new presets, check if name already exists
                    if (CurveAppearancePresetManager::instance()->hasPreset(presetName)) {
                        int result = QMessageBox::question(this, "Preset Exists", 
                                                         QString("Preset '%1' already exists. Overwrite?").arg(presetName),
                                                         QMessageBox::Yes | QMessageBox::No);
                        if (result != QMessageBox::Yes) {
                            return;
                        }
                    }
                }
                
                appearanceWidget->saveCurrentAsPreset(presetName);
            }
        }
    });
    
    // Handle preset delete requests with confirmation dialog
    connect(appearanceWidget, &CurveAppearanceWidget::presetDeleteRequested,
            this, [this, appearanceWidget](const QString &presetName) {
        int result = QMessageBox::question(this, "Delete Preset", 
                                         QString("Delete preset '%1'?").arg(presetName),
                                         QMessageBox::Yes | QMessageBox::No);
        
        if (result == QMessageBox::Yes) {
            appearanceWidget->deletePreset(presetName);
        }
    });
    
    curveWa->setDefaultWidget(appearanceWidget);
    appearanceMenu->addAction(curveWa);
    
    // === COPY/PASTE APPEARANCE GROUP ===
    contextMenu.addSeparator();
    
    // Add Copy Appearance action with copy icon
    QAction *copyAppearanceAction = contextMenu.addAction(QIcon(":/icons/edit-copy.svg"), "Copy Appearance");
    copyAppearanceAction->setToolTip("Copy curve display settings (color, style, thickness, markers, etc.)");
    copyAppearanceAction->setShortcut(QKeySequence("Ctrl+C"));
    connect(copyAppearanceAction, &QAction::triggered, [this, overlay]() {
        copyAppearanceSettings(overlay);
    });
    
    // Add Paste Appearance action with paste icon
    QAction *pasteAppearanceAction = contextMenu.addAction(QIcon(":/icons/edit-paste.svg"), "Paste Appearance");
    pasteAppearanceAction->setToolTip("Paste curve display settings to selected overlays");
    pasteAppearanceAction->setShortcut(QKeySequence("Ctrl+V"));
    pasteAppearanceAction->setEnabled(hasClipboardAppearance());
    connect(pasteAppearanceAction, &QAction::triggered, [this]() {
        pasteAppearanceToSelected();
    });
    
    // === COPY/PASTE SETTINGS GROUP ===
    contextMenu.addSeparator();
    
    // Add Copy Overlay Settings action with properties icon
    QAction *copyAction = contextMenu.addAction(QIcon(":/icons/document-properties.svg"), "Copy Overlay Settings");
    copyAction->setToolTip("Copy overlay properties (plot assignment, scaling, offsets, frequency limits, etc.)");
    copyAction->setShortcut(QKeySequence("Ctrl+Shift+C"));
    connect(copyAction, &QAction::triggered, [this, overlay]() {
        copyOverlaySettings(overlay);
    });
    
    // Add Paste Overlay Settings action with properties paste icon
    QAction *pasteAction = contextMenu.addAction(QIcon(":/icons/document-paste.svg"), "Paste Overlay Settings");
    pasteAction->setToolTip("Paste overlay properties to selected overlays");
    pasteAction->setShortcut(QKeySequence("Ctrl+Shift+V"));
    pasteAction->setEnabled(hasClipboardSettings());
    connect(pasteAction, &QAction::triggered, [this]() {
        pasteSettingsToSelected();
    });
    
    // === UNDO GROUP ===
    contextMenu.addSeparator();
    
    // Add Undo action with undo icon
    QString undoText = getUndoDescription();
    QAction *undoAction = contextMenu.addAction(QIcon(":/icons/edit-undo.svg"), undoText.isEmpty() ? "Undo" : undoText);
    undoAction->setToolTip("Undo the last paste operation");
    undoAction->setShortcut(QKeySequence::Undo);
    undoAction->setEnabled(!undoText.isEmpty());
    connect(undoAction, &QAction::triggered, this, &OverlayManagerWidget::performUndo);
    
    // === OVERLAY MANAGEMENT GROUP ===
    contextMenu.addSeparator();
    
    // Add Remove action with delete icon
    QAction *removeAction = contextMenu.addAction(QIcon(":/icons/remove.png"), "Remove Overlay");
    removeAction->setToolTip("Remove this overlay from the plot");
    connect(removeAction, &QAction::triggered, [this, index]() {
        // Select the row and call remove
        p_overlayTableView->selectRow(index.row());
        removeOverlay();
    });
    
    // Show the context menu at the cursor position
    contextMenu.exec(p_overlayTableView->mapToGlobal(position));
}

void OverlayManagerWidget::copyOverlaySettings(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay) {
        return;
    }
    
    // Clear both clipboards to prevent cross-contamination
    d_clipboardSettings.clear();
    d_clipboardAppearance.clear();
    
    // Copy all overlay settings (excluding label which should remain unique)
    d_clipboardSettings["plotId"] = overlay->getPlotId();
    d_clipboardSettings["yScale"] = overlay->getYScale();
    d_clipboardSettings["yOffset"] = overlay->getYOffset();
    d_clipboardSettings["xOffset"] = overlay->getXOffset();
    d_clipboardSettings["minFreqEnabled"] = overlay->getMinFreqEnabled();
    d_clipboardSettings["minFreqValue"] = overlay->getMinFreqValue();
    d_clipboardSettings["maxFreqEnabled"] = overlay->getMaxFreqEnabled();
    d_clipboardSettings["maxFreqValue"] = overlay->getMaxFreqValue();
    d_clipboardSettings["enabled"] = overlay->getEnabled();
    
    /// TODO: Display message on UI
    qDebug() << "Copied overlay settings from:" << overlay->getLabel();
}

void OverlayManagerWidget::pasteOverlaySettings(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay || d_clipboardSettings.isEmpty()) {
        return;
    }
    
    // Apply copied settings to the target overlay (excluding label)
    if (d_clipboardSettings.contains("plotId")) {
        overlay->setPlotId(d_clipboardSettings["plotId"].toString());
    }
    if (d_clipboardSettings.contains("yScale")) {
        overlay->setYScale(d_clipboardSettings["yScale"].toDouble());
    }
    if (d_clipboardSettings.contains("yOffset")) {
        overlay->setYOffset(d_clipboardSettings["yOffset"].toDouble());
    }
    if (d_clipboardSettings.contains("xOffset")) {
        overlay->setXOffset(d_clipboardSettings["xOffset"].toDouble());
    }
    if (d_clipboardSettings.contains("minFreqEnabled") && d_clipboardSettings.contains("minFreqValue")) {
        overlay->setMinFreqLimit(d_clipboardSettings["minFreqEnabled"].toBool(), 
                                d_clipboardSettings["minFreqValue"].toDouble());
    }
    if (d_clipboardSettings.contains("maxFreqEnabled") && d_clipboardSettings.contains("maxFreqValue")) {
        overlay->setMaxFreqLimit(d_clipboardSettings["maxFreqEnabled"].toBool(), 
                                d_clipboardSettings["maxFreqValue"].toDouble());
    }
    if (d_clipboardSettings.contains("enabled")) {
        overlay->setEnabled(d_clipboardSettings["enabled"].toBool());
    }
    
    // Emit signal to update the overlay display
    emit overlayDataChanged(overlay);
    
    /// TODO: Display message on UI
    qDebug() << "Pasted overlay settings to:" << overlay->getLabel();
}

bool OverlayManagerWidget::hasClipboardSettings() const
{
    return !d_clipboardSettings.isEmpty();
}

void OverlayManagerWidget::copyAppearanceSettings(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay) {
        return;
    }
    
    // Clear both clipboards to prevent cross-contamination
    d_clipboardAppearance.clear();
    d_clipboardSettings.clear();
    
    // Copy curve appearance settings only
    d_clipboardAppearance["curveColor"] = overlay->getCurveMetadata(BC::Key::bcCurveColor);
    d_clipboardAppearance["curveCurveStyle"] = overlay->getCurveMetadata(BC::Key::bcCurveCurveStyle);
    d_clipboardAppearance["curveThickness"] = overlay->getCurveMetadata(BC::Key::bcCurveThickness);
    d_clipboardAppearance["curveLineStyle"] = overlay->getCurveMetadata(BC::Key::bcCurveLineStyle);
    d_clipboardAppearance["curveMarker"] = overlay->getCurveMetadata(BC::Key::bcCurveMarker);
    d_clipboardAppearance["curveMarkerSize"] = overlay->getCurveMetadata(BC::Key::bcCurveMarkerSize);
    d_clipboardAppearance["curveVisible"] = overlay->getCurveMetadata(BC::Key::bcCurveVisible);
    d_clipboardAppearance["curveAutoscale"] = overlay->getCurveMetadata(BC::Key::bcCurveAutoscale);
    d_clipboardAppearance["curveAxisY"] = overlay->getCurveMetadata(BC::Key::bcCurveAxisY);
    
    /// TODO: Display message on UI
    qDebug() << "Copied curve appearance from:" << overlay->getLabel();
}

void OverlayManagerWidget::pasteAppearanceSettings(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay || d_clipboardAppearance.isEmpty()) {
        return;
    }
    
    // Apply copied curve appearance settings
    if (d_clipboardAppearance.contains("curveColor")) {
        overlay->setCurveMetadata(BC::Key::bcCurveColor, d_clipboardAppearance["curveColor"]);
    }
    if (d_clipboardAppearance.contains("curveCurveStyle")) {
        overlay->setCurveMetadata(BC::Key::bcCurveCurveStyle, d_clipboardAppearance["curveCurveStyle"]);
    }
    if (d_clipboardAppearance.contains("curveThickness")) {
        overlay->setCurveMetadata(BC::Key::bcCurveThickness, d_clipboardAppearance["curveThickness"]);
    }
    if (d_clipboardAppearance.contains("curveLineStyle")) {
        overlay->setCurveMetadata(BC::Key::bcCurveLineStyle, d_clipboardAppearance["curveLineStyle"]);
    }
    if (d_clipboardAppearance.contains("curveMarker")) {
        overlay->setCurveMetadata(BC::Key::bcCurveMarker, d_clipboardAppearance["curveMarker"]);
    }
    if (d_clipboardAppearance.contains("curveMarkerSize")) {
        overlay->setCurveMetadata(BC::Key::bcCurveMarkerSize, d_clipboardAppearance["curveMarkerSize"]);
    }
    if (d_clipboardAppearance.contains("curveVisible")) {
        overlay->setCurveMetadata(BC::Key::bcCurveVisible, d_clipboardAppearance["curveVisible"]);
    }
    if (d_clipboardAppearance.contains("curveAutoscale")) {
        overlay->setCurveMetadata(BC::Key::bcCurveAutoscale, d_clipboardAppearance["curveAutoscale"]);
    }
    if (d_clipboardAppearance.contains("curveAxisY")) {
        overlay->setCurveMetadata(BC::Key::bcCurveAxisY, d_clipboardAppearance["curveAxisY"]);
    }
    
    // Emit signal to update the overlay display
    emit overlayDataChanged(overlay);
    
    /// TODO: Display message on UI
    qDebug() << "Pasted curve appearance to:" << overlay->getLabel();
}

bool OverlayManagerWidget::hasClipboardAppearance() const
{
    return !d_clipboardAppearance.isEmpty();
}

void OverlayManagerWidget::onPreviewRequested()
{
    // This slot is called when the creation dialog requests a preview
    // The dialog should have already passed the preview overlay via overlayDataChanged signal
    // Nothing to do here as the preview overlay flows through the normal rendering pipeline
}

void OverlayManagerWidget::onPreviewCancelled()
{
    // This slot is called when the dialog's preview mode is cancelled
    // The actual preview removal is handled by onPreviewOverlayCancelled
}

void OverlayManagerWidget::onPreviewOverlayRequested(std::shared_ptr<OverlayBase> overlay)
{
    if (overlay && p_overlayStorage) {
        // Add preview overlay to storage - this will automatically trigger display
        p_overlayStorage->addPreviewOverlay(overlay);
    }
}

void OverlayManagerWidget::onPreviewOverlayCancelled(std::shared_ptr<OverlayBase> overlay)
{
    if (overlay && p_overlayStorage) {
        // Remove preview overlay from storage - this will automatically trigger removal from plots
        p_overlayStorage->removePreviewOverlay(overlay->getLabel());
    }
}

void OverlayManagerWidget::pasteAppearanceToSelected()
{
    if (!hasClipboardAppearance() || !p_overlayTableView || !p_overlayModel) {
        return;
    }
    
    auto selectionModel = p_overlayTableView->selectionModel();
    if (!selectionModel->hasSelection()) {
        return;
    }
    
    // Get all selected overlays
    QModelIndexList selectedRows = selectionModel->selectedRows();
    QVector<std::shared_ptr<OverlayBase>> selectedOverlays;
    
    for (const auto& index : selectedRows) {
        auto overlay = p_overlayModel->getOverlay(index.row());
        if (overlay) {
            selectedOverlays.append(overlay);
        }
    }
    
    if (selectedOverlays.isEmpty()) {
        return;
    }
    
    // Capture undo state for all selected overlays
    captureUndoState(selectedOverlays, "appearance");
    
    // Apply appearance to all selected overlays
    for (auto overlay : selectedOverlays) {
        pasteAppearanceSettings(overlay);
    }
    
    /// TODO: Display message on UI
    qDebug() << "Pasted appearance to" << selectedOverlays.size() << "overlays";
}

void OverlayManagerWidget::pasteSettingsToSelected()
{
    if (!hasClipboardSettings() || !p_overlayTableView || !p_overlayModel) {
        return;
    }
    
    auto selectionModel = p_overlayTableView->selectionModel();
    if (!selectionModel->hasSelection()) {
        return;
    }
    
    // Get all selected overlays
    QModelIndexList selectedRows = selectionModel->selectedRows();
    QVector<std::shared_ptr<OverlayBase>> selectedOverlays;
    
    for (const auto& index : selectedRows) {
        auto overlay = p_overlayModel->getOverlay(index.row());
        if (overlay) {
            selectedOverlays.append(overlay);
        }
    }
    
    if (selectedOverlays.isEmpty()) {
        return;
    }
    
    // Capture undo state for all selected overlays
    captureUndoState(selectedOverlays, "settings");
    
    // Apply settings to all selected overlays
    for (auto overlay : selectedOverlays) {
        pasteOverlaySettings(overlay);
    }
    
    /// TODO: Display message on UI
    qDebug() << "Pasted settings to" << selectedOverlays.size() << "overlays";
}

void OverlayManagerWidget::closeEvent(QCloseEvent *event)
{
    // Save window geometry
    set(BC::Key::OverlayManager::geometry, saveGeometry(), true);
    
    // Accept the close event
    QWidget::closeEvent(event);
}
