#ifndef OVERLAYMANAGERWIDGET_H
#define OVERLAYMANAGERWIDGET_H

#include <QWidget>
#include <QToolBar>
#include <QAction>
#include <QToolButton>
#include <QMenu>
#include <QVBoxLayout>
#include <QTableView>
#include <QLabel>
#include <QProgressBar>
#include <QShortcut>
#include <memory>

#include <data/experiment/overlaybase.h>
#include <data/model/overlaytablemodel.h>
#include <data/storage/overlaystorage.h>
#include <data/storage/settingsstorage.h>
#include "overlayconfiguredelegate.h"
#include "overlaycheckboxdelegate.h"

#include <gui/plot/curveappearancewidget.h>


namespace BC::Key::OverlayManager {
static const QString key{"OverlayManagerWidget"};
static const QString geometry{"geometry"};
static const QString windowState{"windowState"};
}

class OverlayManagerWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit OverlayManagerWidget(QWidget *parent = nullptr, int number = -1, const QVector<std::shared_ptr<OverlayBase>> &overlays = QVector<std::shared_ptr<OverlayBase>>());
    ~OverlayManagerWidget();

signals:
    void overlayDataChanged(std::shared_ptr<OverlayBase> overlay);

public slots:
    void addOverlay(OverlayBase::OverlayType type);
    void removeOverlay();
    void raiseParent();
    
    // Overlay storage event handlers
    void onOverlayWriteCompleted(std::shared_ptr<OverlayBase> overlay);
    void onOverlayWriteFailed(std::shared_ptr<OverlayBase> overlay, QString error);
    void onPendingWritesChanged(int count);
    
    // Connect to overlay storage signals
    void connectToOverlayStorage(std::shared_ptr<OverlayStorage> storage);
    
    // Handle external overlay data changes (e.g., from plot context menu)
    void onExternalOverlayDataChanged(std::shared_ptr<OverlayBase> overlay);

private slots:
    void onModelDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);
    void onSelectionChanged();
    void onConfigureClicked(const QModelIndex &index);
    void onOverlaySettingsChanged(std::shared_ptr<OverlayBase> overlay);
    void showContextMenu(const QPoint &position);
    void copyOverlaySettings(std::shared_ptr<OverlayBase> overlay);
    void pasteOverlaySettings(std::shared_ptr<OverlayBase> overlay);
    void copyAppearanceSettings(std::shared_ptr<OverlayBase> overlay);
    void pasteAppearanceSettings(std::shared_ptr<OverlayBase> overlay);
    void pasteAppearanceToSelected();
    void pasteSettingsToSelected();
    bool hasClipboardSettings() const;
    bool hasClipboardAppearance() const;

private:
    QToolBar *p_toolBar;
    QToolButton *p_addButton;
    QMenu *p_addMenu;
    std::map<OverlayBase::OverlayType, QAction*> d_addActions;
    QAction *p_removeAction;
    QAction *p_raiseParentAction;

    OverlayTableModel *p_overlayModel;
    QTableView *p_overlayTableView;
    OverlayConfigureDelegate *p_configureDelegate;
    OverlayCheckBoxDelegate *p_enabledDelegate;
    
    // Keyboard shortcuts
    QShortcut *p_copyAppearanceShortcut;
    QShortcut *p_pasteAppearanceShortcut;
    QShortcut *p_copySettingsShortcut;
    QShortcut *p_pasteSettingsShortcut;
    QShortcut *p_undoShortcut;
    
    // Progress indicator widgets
    QLabel *p_progressLabel;
    QProgressBar *p_progressBar;
    QWidget *p_progressWidget;
    
    // Track overlay storage connection
    std::shared_ptr<OverlayStorage> p_overlayStorage;
    
    // Clipboard for copy/paste overlay settings
    QVariantMap d_clipboardSettings;
    
    // Clipboard for copy/paste curve appearance settings
    QVariantMap d_clipboardAppearance;
    
    // Undo system for paste operations (1 level deep)
    struct UndoData {
        bool hasUndoData = false;
        std::shared_ptr<OverlayBase> overlay = nullptr;
        QString operationType; // "appearance" or "settings" 
        QVariantMap previousAppearanceData;
        QVariantMap previousSettingsData;
    };
    UndoData d_undoData;


    void setupUI();
    void setupAddButton();
    void updateButtonStates();
    void populateWithExistingOverlays(const QVector<std::shared_ptr<OverlayBase>> &overlays);
    void setupConfigureDelegate();
    void setupEnabledDelegate();
    void setupTableView();
    void setupKeyboardShortcuts();
    void resizeColumnsToContents();
    
    // Selection information structure
    struct SelectionInfo {
        bool singleRowSelected = false;
        bool multipleRowsSelected = false;
        int selectedCount = 0;
        std::shared_ptr<OverlayBase> overlay = nullptr; // Only valid when singleRowSelected is true
    };
    
    SelectionInfo getSelectionInfo();
    std::shared_ptr<OverlayBase> getSelectedOverlay(); // Keep for backward compatibility
    
    // Undo system methods
    void captureUndoState(std::shared_ptr<OverlayBase> overlay, const QString &operationType);
    void performUndo();
    void invalidateUndo();
    
    // Progress indicator management
    void createProgressWidget();
    void updateProgressDisplay(int pendingCount);
    void showErrorNotification(const QString& overlayLabel, const QString& error);
    
    
protected:
    void closeEvent(QCloseEvent *event) override;
};

#endif // OVERLAYMANAGERWIDGET_H
