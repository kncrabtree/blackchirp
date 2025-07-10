#include "overlaymanagerwidget.h"
#include "bcexpoverlaydialog.h"
#include "overlayconfiguredelegate.h"
#include "overlaysettingsdialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QDebug>
#include <QTableView>
#include <QHeaderView>
#include <QMessageBox>
#include <QProgressBar>
#include <QCloseEvent>
#include <gui/widget/ftmwviewwidget.h>

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
        case OverlayBase::SPCAT:
            menuItemName = "SPCAT Catalog";
            enabled = false; // Not yet implemented
            break;
        case OverlayBase::GenericXY:
            menuItemName = "Generic XY Data";
            enabled = false; // Not yet implemented
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

    // pointers used for the overlay dialog and overlay object
    std::shared_ptr<OverlayBase> overlay = nullptr;
    OverlayConfigDialog *dialog = nullptr;

    // Get the FtmwViewWidget parent for dialog constructors
    FtmwViewWidget* ftmwParent = qobject_cast<FtmwViewWidget*>(parentWidget());
    if(!ftmwParent) {
        return;
    }
    
    // Create appropriate dialog based on overlay type
    switch(type) {
    case OverlayBase::BCExperiment:
        {
            dialog = new BCExpOverlayDialog(ftmwParent);
            break;
        }
    case OverlayBase::SPCAT:
        // TODO: Implement SPCAT overlay creation
        // dialog = new SPCATOverlayDialog(ftmwParent);
        qDebug() << "SPCAT overlay creation not yet implemented";
        return;
    case OverlayBase::GenericXY:
        // TODO: Implement GenericXY overlay creation
        // dialog = new GenericXYOverlayDialog(ftmwParent);
        qDebug() << "GenericXY overlay creation not yet implemented";
        return;
    default:
        qDebug() << "Unknown overlay type";
        return;
    }

    // Run the dialog and get the overlay if accepted
    dialog->setModal(true);
    dialog->setupUI(); // Set up UI after construction is complete
    if(dialog->exec() == QDialog::Accepted) {
        // Create the overlay
        overlay = dialog->createOverlay();
    }
    
    // Add overlay to storage if created successfully
    if (overlay != nullptr) {
        // Add directly to overlay storage - this initiates async write
        if (p_overlayStorage->addOverlay(overlay)) {
            // Add to unified model for display
            p_overlayModel->addOverlay(overlay);
            
            // Update UI state to show any pending writes
            updateButtonStates();
        }
    }

    dialog->deleteLater();
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
                
                qDebug() << "Overlay removed from storage successfully";
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
    
    // Get xRange from the main plot and plot names
    auto xRange = ftmwParent->getMainPlotFt().xRange();
    QStringList plotNames = ftmwParent->getPlotNames();
    
    // Create and show the configuration dialog
    OverlaySettingsDialog dialog(overlay, plotNames, xRange.first, xRange.second, p_overlayStorage, this);
    dialog.setupUI(); // Set up UI after construction is complete
    
    // Connect the dialog signal to our slot for real-time updates
    connect(&dialog, &OverlaySettingsDialog::overlaySettingsChanged,
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
    
    // Add Configure action
    QAction *configureAction = contextMenu.addAction("Configure...");
    connect(configureAction, &QAction::triggered, [this, index]() {
        onConfigureClicked(index);
    });
    
    contextMenu.addSeparator();
    
    // Add Copy Settings action
    QAction *copyAction = contextMenu.addAction("Copy Settings");
    connect(copyAction, &QAction::triggered, [this, overlay]() {
        copyOverlaySettings(overlay);
    });
    
    // Add Paste Settings action (only enable if we have copied settings)
    QAction *pasteAction = contextMenu.addAction("Paste Settings");
    pasteAction->setEnabled(hasClipboardSettings());
    connect(pasteAction, &QAction::triggered, [this, overlay]() {
        pasteOverlaySettings(overlay);
    });
    
    contextMenu.addSeparator();
    
    // Add Remove action
    QAction *removeAction = contextMenu.addAction("Remove");
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
    
    // Clear previous clipboard contents
    d_clipboardSettings.clear();
    
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
    
    qDebug() << "Pasted overlay settings to:" << overlay->getLabel();
}

bool OverlayManagerWidget::hasClipboardSettings() const
{
    return !d_clipboardSettings.isEmpty();
}

void OverlayManagerWidget::closeEvent(QCloseEvent *event)
{
    // Save window geometry
    set(BC::Key::OverlayManager::geometry, saveGeometry(), true);
    
    // Accept the close event
    QWidget::closeEvent(event);
}
