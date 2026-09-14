/**
 * ================================================================
 * AI DATA ANALYST — COMPLETE script.js
 * ================================================================
 *
 * FLOW:
 *
 * Upload
 *   ↓
 * /upload
 *   ↓
 * Dashboard + Dataset
 *
 * AI Analyst
 *   ↓
 * /chat
 *   ↓
 * agent.py calculates result
 *   ↓
 * Answer is shown ONLY in chat
 *   ↓
 * Precomputed result is stored silently
 *
 * User clicks Visualization Studio
 *   ↓
 * Exact stored result is sent to /visualize/analysis
 *   ↓
 * Chart appears ONLY in Visualization Studio
 *
 * The frontend NEVER recalculates an AI question.
 * ================================================================
 */

document.addEventListener("DOMContentLoaded", () => {

    /* ============================================================
       GLOBAL STATE
       ============================================================ */

    let currentAgentResult = null;
    let visualizationResultLoading = false;

    let mediaRecorder = null;
    let audioChunks = [];
    let recordingTimer = null;
    let recordingSeconds = 0;


    /* ============================================================
       ELEMENT HELPER
       ============================================================ */

    const $ = id => document.getElementById(id);


    /* ============================================================
       ELEMENTS
       ============================================================ */

    const navButtons =
        document.querySelectorAll(".nav-btn");

    const viewSections =
        document.querySelectorAll(".view-section");

    const uploadForm =
        $("upload-form");

    const fileInput =
        $("file-input");

    const filenameDisplay =
        $("upload-filename-display");

    const loader =
        $("global-loader");

    const loaderText =
        $("loader-text");

    const chatForm =
        $("chat-form");

    const chatInput =
        $("chat-input");

    const chatMessages =
        $("chat-messages");

    const clearChatBtn =
        $("clear-chat-btn");

    const renderChartBtn =
        $("render-chart-btn");

    const micBtn =
        $("mic-btn");

    const stopRecordingBtn =
        $("stop-recording-btn");

    const voiceRecordingBar =
        $("voice-recording-bar");

    const voiceTimerEl =
        $("voice-timer");


    /* ============================================================
       1. NAVIGATION
       ============================================================ */

    navButtons.forEach(btn => {

        btn.addEventListener("click", () => {

            const targetView =
                btn.getAttribute("data-view");

            if (!targetView) {
                return;
            }

            navButtons.forEach(button => {
                button.classList.remove("active");
            });

            viewSections.forEach(section => {
                section.classList.remove("active");
            });

            btn.classList.add("active");

            const target =
                $(targetView);

            if (target) {
                target.classList.add("active");
            }


            /*
             * Profile
             */
            if (targetView === "profile-view") {
                fetchDatasetProfile();
            }


            /*
             * Quality
             */
            if (targetView === "quality-view") {
                fetchQualityReport();
            }


            /*
             * IMPORTANT:
             *
             * Visualization is ONLY generated when the
             * user explicitly opens Visualization Studio.
             *
             * AI Analyst NEVER calls this automatically.
             */
            if (
                targetView ===
                "visualizations-view"
            ) {

                renderStoredAgentResultWhenOpened();

            }

        });

    });


    /* ============================================================
       2. FILE UPLOAD
       ============================================================ */

    if (fileInput) {

        fileInput.addEventListener(
            "change",
            () => {

                if (
                    fileInput.files &&
                    fileInput.files.length > 0
                ) {

                    const file =
                        fileInput.files[0];

                    if (filenameDisplay) {

                        filenameDisplay.textContent =
                            `Selected: ${file.name}`;

                    }

                } else {

                    if (filenameDisplay) {
                        filenameDisplay.textContent = "";
                    }

                }

            }
        );

    }


    if (uploadForm) {

        uploadForm.addEventListener(
            "submit",
            async event => {

                event.preventDefault();


                if (
                    !fileInput ||
                    !fileInput.files ||
                    fileInput.files.length === 0
                ) {

                    alert(
                        "Please select a file to upload first."
                    );

                    return;
                }


                const file =
                    fileInput.files[0];


                /*
                 * Supported files.
                 */
                const allowed =
                    [
                        ".csv",
                        ".xlsx",
                        ".xls"
                    ];


                const lowerName =
                    file.name.toLowerCase();


                const valid =
                    allowed.some(
                        extension =>
                            lowerName.endsWith(
                                extension
                            )
                    );


                if (!valid) {

                    alert(
                        "Please select a CSV, XLSX, or XLS file."
                    );

                    return;
                }


                /*
                 * Flask backend expects:
                 *
                 * request.files["dataset"]
                 */
                const formData =
                    new FormData();


                formData.append(
                    "dataset",
                    file
                );


                showLoader(
                    "Uploading & analyzing dataset..."
                );


                try {

                    const response =
                        await fetch(
                            "/upload",
                            {
                                method: "POST",
                                body: formData
                            }
                        );


                    const text =
                        await response.text();


                    let data;


                    try {

                        data =
                            JSON.parse(text);

                    } catch (error) {

                        console.error(
                            "Non-JSON upload response:",
                            text
                        );

                        alert(
                            `Upload failed (${response.status}): ${text.slice(0, 300) ||
                            response.statusText ||
                            "Server error"
                            }`
                        );

                        return;
                    }


                    if (
                        response.ok &&
                        data
                    ) {

                        /*
                         * Filename badge.
                         */
                        const badge =
                            $("badge-filename");

                        const activeBadge =
                            $("active-file-badge");


                        if (badge) {

                            badge.textContent =
                                data.filename ||
                                file.name;

                        }


                        if (activeBadge) {

                            activeBadge.classList.remove(
                                "hidden"
                            );

                        }


                        /*
                         * Filename below Choose File.
                         */
                        if (filenameDisplay) {

                            filenameDisplay.textContent =
                                `Selected: ${data.filename ||
                                file.name
                                }`;

                        }


                        /*
                         * Dashboard metrics.
                         */
                        if (data.metrics) {

                            renderDashboardMetrics(
                                data.metrics
                            );

                        }


                        /*
                         * Refresh dataset.
                         */
                        await fetchDatasetInfo();


                        /*
                         * Refresh AI suggestions.
                         */
                        await fetchSuggestions();


                        /*
                         * Old result belongs to old dataset.
                         */
                        currentAgentResult =
                            null;


                    } else {

                        alert(
                            (
                                data &&
                                data.error
                            ) ||
                            `Upload failed with status ${response.status}.`
                        );

                    }


                } catch (error) {

                    console.error(
                        "Upload error:",
                        error
                    );


                    alert(
                        "Upload network error: " +
                        (
                            error.message ||
                            "Please check server connection."
                        )
                    );


                } finally {

                    hideLoader();

                }

            }
        );

    }


    /* ============================================================
       3. DATASET INFORMATION
       ============================================================ */

    async function fetchDatasetInfo() {

        try {

            const response =
                await fetch(
                    "/dataset"
                );


            if (!response.ok) {
                return;
            }


            const data =
                await response.json();


            const emptyState =
                $("empty-dashboard-state");


            const dashboardContent =
                $("dashboard-content");


            if (emptyState) {

                emptyState.classList.add(
                    "hidden"
                );

            }


            if (dashboardContent) {

                dashboardContent.classList.remove(
                    "hidden"
                );

            }


            if (data.preview) {

                renderPreviewTable(
                    data.preview
                );

            }


            populateAxisDropdowns(
                data.columns || [],
                data.numeric_columns || []
            );


        } catch (error) {

            console.warn(
                "Dataset information unavailable:",
                error
            );

        }

    }


    /* ============================================================
       4. DASHBOARD METRICS
       ============================================================ */

    function renderDashboardMetrics(metrics) {

        if (!metrics) {
            return;
        }


        const rows =
            $("m-rows");

        const cols =
            $("m-cols");

        const numeric =
            $("m-numeric");

        const missing =
            $("m-missing");

        const duplicates =
            $("m-duplicates");


        if (rows) {

            rows.textContent =
                Number(
                    metrics.total_rows || 0
                ).toLocaleString();

        }


        if (cols) {

            cols.textContent =
                metrics.total_columns ?? 0;

        }


        if (numeric) {

            numeric.textContent =
                metrics.numeric_columns_count ?? 0;

        }


        if (missing) {

            missing.textContent =
                Number(
                    metrics.missing_cells || 0
                ).toLocaleString();

        }


        if (duplicates) {

            duplicates.textContent =
                Number(
                    metrics.duplicate_rows || 0
                ).toLocaleString();

        }

    }


    /* ============================================================
       5. DATA PREVIEW
       ============================================================ */

    function renderPreviewTable(records) {

        if (
            !Array.isArray(records) ||
            records.length === 0
        ) {
            return;
        }


        const table =
            $("preview-table");


        if (!table) {
            return;
        }


        const thead =
            table.querySelector("thead");

        const tbody =
            table.querySelector("tbody");


        if (!thead || !tbody) {
            return;
        }


        const headers =
            Object.keys(records[0]);


        thead.innerHTML =
            `
                <tr>
                    ${headers
                .map(
                    header =>
                        `<th>${escapeHtml(header)}</th>`
                )
                .join("")
            }
                </tr>
            `;


        tbody.innerHTML =
            records
                .map(row => {

                    return `
                        <tr>
                            ${headers
                            .map(header => {

                                const value =
                                    row[header];

                                return `
                                            <td>
                                                ${value === null ||
                                        value === undefined
                                        ? ""
                                        : escapeHtml(
                                            String(value)
                                        )
                                    }
                                            </td>
                                        `;

                            })
                            .join("")
                        }
                        </tr>
                    `;

                })
                .join("");

    }


    /* ============================================================
       6. AXIS DROPDOWNS
       ============================================================ */

    function populateAxisDropdowns(
        allColumns,
        numericColumns
    ) {

        const xSelect =
            $("x-axis-select");

        const ySelect =
            $("y-axis-select");


        if (xSelect) {

            xSelect.innerHTML = "";


            allColumns.forEach(
                column => {

                    const option =
                        document.createElement(
                            "option"
                        );

                    option.value =
                        column;

                    option.textContent =
                        column;

                    xSelect.appendChild(
                        option
                    );

                }
            );

        }


        if (ySelect) {

            ySelect.innerHTML = "";


            numericColumns.forEach(
                column => {

                    const option =
                        document.createElement(
                            "option"
                        );

                    option.value =
                        column;

                    option.textContent =
                        column;

                    ySelect.appendChild(
                        option
                    );

                }
            );

        }

    }


    /*
     * Agent-computed columns such as:
     *
     * Average_Order_Value
     * Average_Order_Profit
     * Average_Profit_Ratio
     *
     * may not exist in the original dataset.
     *
     * They are therefore inserted dynamically.
     */
    function ensureSelectOption(
        select,
        value
    ) {

        if (
            !select ||
            value === null ||
            value === undefined
        ) {
            return;
        }


        const stringValue =
            String(value);


        const exists =
            Array.from(
                select.options
            ).some(
                option =>
                    option.value ===
                    stringValue
            );


        if (!exists) {

            const option =
                document.createElement(
                    "option"
                );


            option.value =
                stringValue;


            option.textContent =
                stringValue;


            select.appendChild(
                option
            );

        }


        select.value =
            stringValue;

    }


    /* ============================================================
       7. AI ANALYST
       ============================================================ */

    if (chatForm) {

        chatForm.addEventListener(
            "submit",
            async event => {

                event.preventDefault();


                const question =
                    chatInput
                        ? chatInput.value.trim()
                        : "";


                if (!question) {
                    return;
                }


                appendChatMessage(
                    "user",
                    question
                );


                if (chatInput) {
                    chatInput.value = "";
                }


                showLoader(
                    "AI Agent thinking..."
                );


                try {

                    const response =
                        await fetch(
                            "/chat",
                            {
                                method: "POST",

                                headers: {
                                    "Content-Type":
                                        "application/json"
                                },

                                /*
                                 * Send both keys so this frontend
                                 * works with either backend version.
                                 */
                                body: JSON.stringify({

                                    message:
                                        question,

                                    question:
                                        question

                                })
                            }
                        );


                    const data =
                        await response.json();


                    if (!response.ok) {

                        appendChatMessage(
                            "assistant",
                            `⚠️ ${data.error ||
                            "Unable to process the question."
                            }`
                        );

                        return;
                    }


                    /*
                     * ====================================================
                     * IMPORTANT:
                     *
                     * ONLY ANSWER IS SHOWN HERE.
                     *
                     * NO CHART.
                     * NO VISUALIZATION NAVIGATION.
                     * NO CHART RENDERING.
                     * ====================================================
                     */
                    appendChatMessage(
                        "assistant",
                        data.answer ||
                        "No answer returned."
                    );


                    /*
                     * Store the exact result silently.
                     *
                     * This is NOT displayed in AI Analyst.
                     */
                    if (
                        String(
                            data.calculation_status ||
                            "calculated"
                        ).toLowerCase() !==
                        "unavailable" &&

                        Array.isArray(
                            data.result_data
                        ) &&

                        data.result_data.length > 0 &&

                        data.x_col &&

                        data.y_col
                    ) {

                        currentAgentResult =
                            data;

                    } else {

                        currentAgentResult =
                            null;

                    }


                } catch (error) {

                    console.error(
                        "Chat error:",
                        error
                    );


                    appendChatMessage(
                        "assistant",
                        "⚠️ Server connection failure. Please try again."
                    );


                } finally {

                    hideLoader();

                }

            }
        );

    }


    /* ============================================================
       8. CHAT MARKDOWN
       ============================================================ */

    function formatMarkdown(text) {

        if (
            text === null ||
            text === undefined
        ) {
            return "";
        }


        let safe =
            escapeHtml(
                String(text)
            );


        /*
         * Bullets.
         */
        safe =
            safe.replace(
                /((?:^[ \t]*[-*][ \t]+.+\n?)+)/gm,
                block => {

                    const items =
                        block
                            .split("\n")
                            .filter(
                                line =>
                                    line.trim()
                            )
                            .map(
                                line =>
                                    `<li>${line.replace(
                                        /^[ \t]*[-*][ \t]+/,
                                        ""
                                    )
                                    }</li>`
                            )
                            .join("");


                    return `<ul>${items}</ul>`;

                }
            );


        /*
         * Bold.
         */
        safe =
            safe.replace(
                /\*\*(.+?)\*\*/g,
                "<strong>$1</strong>"
            );


        /*
         * Italic.
         */
        safe =
            safe.replace(
                /\*([^*\n]+?)\*/g,
                "<em>$1</em>"
            );


        /*
         * Code.
         */
        safe =
            safe.replace(
                /`([^`]+?)`/g,
                "<code>$1</code>"
            );


        /*
         * Newlines.
         */
        safe =
            safe.replace(
                /\n/g,
                "<br>"
            );


        return safe;

    }


    function appendChatMessage(
        role,
        text
    ) {

        if (!chatMessages) {
            return;
        }


        const message =
            document.createElement(
                "div"
            );


        message.className =
            `message ${role}-message`;


        const formatted =
            role === "assistant"
                ? formatMarkdown(text)
                : escapeHtml(text);


        message.innerHTML = `

            <div class="avatar">
                ${role === "user"
                ? "<i class='fa-solid fa-user'></i>"
                : "<i class='fa-solid fa-robot'></i>"
            }
            </div>

            <div class="message-content">
                ${formatted}
            </div>

        `;


        chatMessages.appendChild(
            message
        );


        chatMessages.scrollTop =
            chatMessages.scrollHeight;

    }


    /*
     * This function is kept for compatibility,
     * but AI Analyst no longer calls it.
     */
    function appendChatChart(chartJson) {

        if (
            !chartJson ||
            typeof Plotly ===
            "undefined" ||
            !chatMessages
        ) {
            return;
        }


        const chartId =
            `chat-chart-${Date.now()}-${Math.floor(
                Math.random() * 10000
            )}`;


        const message =
            document.createElement(
                "div"
            );


        message.className =
            "message assistant-message";


        message.innerHTML = `

            <div class="avatar">
                📊
            </div>

            <div
                class="message-content"
                style="width:100%;"
            >

                <div
                    id="${chartId}"
                    style="
                        height:300px;
                        width:100%;
                    "
                ></div>

            </div>

        `;


        chatMessages.appendChild(
            message
        );


        const chart =
            chartJson.chart ||
            chartJson;


        if (!chart.data) {
            return;
        }


        Plotly.newPlot(
            chartId,
            chart.data,
            chart.layout || {},
            {
                responsive: true,
                displayModeBar: false
            }
        );


        chatMessages.scrollTop =
            chatMessages.scrollHeight;

    }


    /* ============================================================
       9. CLEAR CHAT
       ============================================================ */

    if (clearChatBtn) {

        clearChatBtn.addEventListener(
            "click",
            () => {

                if (!chatMessages) {
                    return;
                }


                chatMessages.innerHTML = `

                    <div
                        class="message assistant-message"
                    >

                        <div class="avatar">
                            🤖
                        </div>

                        <div class="message-content">

                            <p>
                                Chat history cleared.
                                How else can I assist?
                            </p>

                        </div>

                    </div>

                `;

            }
        );

    }


    /* ============================================================
       10. SUGGESTIONS
       ============================================================ */

    async function fetchSuggestions() {

        try {

            const response =
                await fetch(
                    "/suggestions"
                );


            if (!response.ok) {
                return;
            }


            const data =
                await response.json();


            const container =
                $("suggestions-container");


            if (!container) {
                return;
            }


            if (
                Array.isArray(
                    data.suggestions
                ) &&
                data.suggestions.length
            ) {

                container.innerHTML =
                    data.suggestions
                        .map(
                            suggestion => {

                                const question =
                                    suggestion.question ||
                                    suggestion;


                                return `

                                    <div
                                        class="chip"
                                        data-q="${escapeHtml(
                                    question
                                )}"
                                    >
                                        ${escapeHtml(
                                    question
                                )}
                                    </div>

                                `;

                            }
                        )
                        .join("");


                container
                    .querySelectorAll(
                        ".chip"
                    )
                    .forEach(chip => {

                        chip.addEventListener(
                            "click",
                            () => {

                                if (!chatInput) {
                                    return;
                                }


                                chatInput.value =
                                    chip.getAttribute(
                                        "data-q"
                                    ) || "";


                                if (chatForm) {

                                    chatForm.dispatchEvent(
                                        new Event(
                                            "submit",
                                            {
                                                bubbles: true,
                                                cancelable: true
                                            }
                                        )
                                    );

                                }

                            }
                        );

                    });

            }


        } catch (error) {

            console.warn(
                "Suggestions error:",
                error
            );

        }

    }


    /* ============================================================
       11. VISUALIZATION STUDIO
       ============================================================ */

    /*
     * This is the ONLY place where an AI result is rendered.
     *
     * It is triggered by:
     *
     * User → clicks Visualization Studio
     */
    async function renderStoredAgentResultWhenOpened() {

        const agentResult =
            currentAgentResult;


        if (!agentResult) {
            return;
        }


        if (visualizationResultLoading) {
            return;
        }


        if (
            String(
                agentResult.calculation_status ||
                "calculated"
            ).toLowerCase() ===
            "unavailable"
        ) {
            return;
        }


        if (
            !Array.isArray(
                agentResult.result_data
            ) ||
            agentResult.result_data.length === 0 ||
            !agentResult.x_col ||
            !agentResult.y_col
        ) {
            return;
        }


        visualizationResultLoading =
            true;


        try {

            /*
             * Populate EXACT agent-selected fields.
             */
            setVisualizationSelection(
                agentResult.x_col,
                agentResult.y_col,
                agentResult.chart_type ||
                "bar"
            );


            /*
             * IMPORTANT:
             *
             * Send result_data exactly as returned by
             * agent.py.
             *
             * Do NOT send the raw dataset.
             */
            const response =
                await fetch(
                    "/visualize/analysis",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({

                            chart_type:
                                agentResult.chart_type ||
                                "bar",

                            x_col:
                                agentResult.x_col,

                            y_col:
                                agentResult.y_col,

                            title:
                                agentResult.chart_title ||
                                "AI Analysis Result",

                            chart_title:
                                agentResult.chart_title ||
                                "AI Analysis Result",

                            result_data:
                                agentResult.result_data,

                            calculation_status:
                                agentResult.calculation_status ||
                                "calculated",

                            reason:
                                agentResult.reason ||
                                null

                        })
                    }
                );


            const data =
                await response.json();


            if (
                !response.ok
            ) {

                console.error(
                    "Visualization error:",
                    data
                );

                return;
            }


            if (
                String(
                    data.calculation_status ||
                    "calculated"
                ).toLowerCase() ===
                "unavailable"
            ) {
                return;
            }


            if (!data.chart) {

                console.error(
                    "No chart returned."
                );

                return;
            }


            /*
             * Render chart ONLY inside Visualization Studio.
             */
            renderVisualizationChart(
                data.chart
            );


        } catch (error) {

            console.error(
                "Stored agent visualization error:",
                error
            );

        } finally {

            visualizationResultLoading =
                false;

        }

    }


    /*
     * Compatibility function.
     *
     * IMPORTANT:
     *
     * Nothing calls this from AI Analyst anymore.
     *
     * It exists only in case another part of the
     * application calls it.
     */
    async function syncAgentResultToVisualization(
        agentResult
    ) {

        if (!agentResult) {
            return;
        }


        currentAgentResult =
            agentResult;


        /*
         * Do NOT open Visualization Studio.
         * Do NOT render chart.
         *
         * The user must click Visualization Studio.
         */

    }


    /* ============================================================
       12. SET VISUALIZATION SELECTION
       ============================================================ */

    function setVisualizationSelection(
        xColumn,
        yColumn,
        chartType
    ) {

        const xSelect =
            $("x-axis-select");

        const ySelect =
            $("y-axis-select");

        const chartSelect =
            $("chart-type-select");


        /*
         * X axis.
         */
        if (
            xSelect &&
            xColumn
        ) {

            ensureSelectOption(
                xSelect,
                xColumn
            );


            xSelect.value =
                String(xColumn);

        }


        /*
         * Y axis.
         */
        if (
            ySelect &&
            yColumn
        ) {

            ensureSelectOption(
                ySelect,
                yColumn
            );


            ySelect.value =
                String(yColumn);

        }


        /*
         * Chart type.
         */
        if (
            chartSelect &&
            chartType
        ) {

            const normalized =
                normalizeChartType(
                    chartType
                );


            ensureSelectOption(
                chartSelect,
                normalized
            );


            chartSelect.value =
                normalized;

        }

    }


    function normalizeChartType(
        chartType
    ) {

        if (!chartType) {
            return "bar";
        }


        const type =
            String(
                chartType
            )
                .toLowerCase()
                .trim();


        const aliases = {

            column:
                "bar",

            columns:
                "bar",

            bar_chart:
                "bar",

            bargraph:
                "bar",

            bar_graph:
                "bar",

            line_chart:
                "line",

            pie_chart:
                "pie",

            donut:
                "pie",

            doughnut:
                "pie",

            scatter_plot:
                "scatter",

            scatterplot:
                "scatter"

        };


        return (
            aliases[type] ||
            type
        );

    }


    /* ============================================================
       13. RENDER PLOTLY CHART
       ============================================================ */

    function renderVisualizationChart(
        chartJson
    ) {

        if (
            !chartJson ||
            typeof Plotly ===
            "undefined"
        ) {

            console.error(
                "Plotly is unavailable."
            );

            return;
        }


        const box =
            $("plotly-chart-container");


        if (!box) {

            console.error(
                "#plotly-chart-container not found."
            );

            return;
        }


        const chart =
            chartJson.chart ||
            chartJson;


        if (
            !chart ||
            !chart.data
        ) {

            console.error(
                "Invalid Plotly chart payload."
            );

            return;
        }


        box.innerHTML = "";


        const layout = {

            ...(chart.layout || {}),

            autosize:
                true,

            width:
                null,

            height:
                chart.layout?.height ||
                460

        };


        Plotly.newPlot(

            box,

            chart.data,

            layout,

            {

                responsive:
                    true,

                useResizeHandler:
                    true

            }

        ).then(() => {

            try {

                Plotly.Plots.resize(
                    box
                );

            } catch (error) { }

        });

    }


    /* ============================================================
       14. MANUAL VISUALIZATION
       ============================================================ */

    if (renderChartBtn) {

        renderChartBtn.addEventListener(
            "click",
            async () => {

                const chartType =
                    $("chart-type-select")?.value;


                const xCol =
                    $("x-axis-select")?.value;


                const yCol =
                    $("y-axis-select")?.value;


                if (
                    !chartType ||
                    !xCol ||
                    !yCol
                ) {

                    alert(
                        "Please select Chart Type, X Axis, and Y Axis."
                    );

                    return;
                }


                showLoader(
                    "Building chart visual..."
                );


                try {

                    const response =
                        await fetch(
                            "/visualize",
                            {
                                method: "POST",

                                headers: {
                                    "Content-Type":
                                        "application/json"
                                },

                                body:
                                    JSON.stringify({

                                        chart_type:
                                            chartType,

                                        x_col:
                                            xCol,

                                        y_col:
                                            yCol

                                    })

                            }
                        );


                    const data =
                        await response.json();


                    if (!response.ok) {

                        alert(
                            data.error ||
                            "Chart rendering failed."
                        );

                        return;
                    }


                    if (
                        String(
                            data.calculation_status ||
                            "calculated"
                        ).toLowerCase() ===
                        "unavailable"
                    ) {

                        alert(
                            data.reason ||
                            data.error ||
                            "This visualization cannot be calculated."
                        );

                        return;
                    }


                    if (data.chart) {

                        renderVisualizationChart(
                            data.chart
                        );

                    }


                } catch (error) {

                    console.error(
                        "Manual visualization error:",
                        error
                    );


                    alert(
                        "Chart rendering failed."
                    );


                } finally {

                    hideLoader();

                }

            }
        );

    }


    /* ============================================================
       15. INSIGHTS
       ============================================================ */

    const generateInsightsBtn =
        $("generate-insights-btn");


    if (generateInsightsBtn) {

        generateInsightsBtn.addEventListener(
            "click",
            fetchInsights
        );

    }


    async function fetchInsights() {

        showLoader(
            "Generating automated AI insights..."
        );


        try {

            const response =
                await fetch(
                    "/insights"
                );


            const data =
                await response.json();


            const container =
                $("insights-container");


            if (!container) {
                return;
            }


            if (
                response.ok &&
                Array.isArray(
                    data.insights
                ) &&
                data.insights.length > 0
            ) {

                container.innerHTML =
                    data.insights
                        .map(
                            insight => `

                                <div
                                    class="insight-card"
                                >

                                    <div
                                        class="insight-header"
                                    >

                                        <h4
                                            class="insight-title"
                                        >
                                            ${escapeHtml(
                                insight.title ||
                                ""
                            )}
                                        </h4>


                                        <span
                                            class="badge ${insight.level ===
                                    "Good"
                                    ? "badge-good"
                                    : "badge-warning"
                                }"
                                        >
                                            ${escapeHtml(
                                    insight.level ||
                                    ""
                                )}
                                        </span>

                                    </div>


                                    <p
                                        style="
                                            color:#64748B;
                                            font-size:.875rem;
                                            margin-top:.5rem;
                                        "
                                    >
                                        ${escapeHtml(
                                    insight.explanation ||
                                    ""
                                )}
                                    </p>


                                    <div
                                        style="
                                            margin-top:.75rem;
                                            font-weight:600;
                                            font-size:.85rem;
                                            color:#2563EB;
                                        "
                                    >
                                        Indicator:
                                        ${escapeHtml(
                                    insight.metric ||
                                    ""
                                )}
                                    </div>

                                </div>

                            `
                        )
                        .join("");

            } else {

                container.innerHTML = `

                    <div
                        class="empty-state"
                    >

                        <p
                            style="
                                color:#EF4444;
                            "
                        >
                            ${escapeHtml(
                    data.error ||
                    "Please upload a dataset first to generate AI insights."
                )}
                        </p>

                    </div>

                `;

            }


        } catch (error) {

            console.error(
                "Insights error:",
                error
            );


            alert(
                "Failed to reach server endpoint for insights."
            );


        } finally {

            hideLoader();

        }

    }


    /* ============================================================
       16. PROFILE
       ============================================================ */

    async function fetchDatasetProfile() {

        try {

            const response =
                await fetch(
                    "/profile"
                );


            if (!response.ok) {
                return;
            }


            const data =
                await response.json();


            const table =
                $("profile-table");


            if (!table) {
                return;
            }


            const tbody =
                table.querySelector(
                    "tbody"
                );


            if (!tbody) {
                return;
            }


            tbody.innerHTML =
                (
                    data.profiles || []
                )
                    .map(
                        profile => `

                            <tr>

                                <td>
                                    <strong>
                                        ${escapeHtml(
                            profile.column_name
                        )}
                                    </strong>
                                </td>

                                <td>
                                    <code>
                                        ${escapeHtml(
                            profile.data_type
                        )}
                                    </code>
                                </td>

                                <td>
                                    ${escapeHtml(
                            String(
                                profile.unique_values ??
                                ""
                            )
                        )}
                                </td>

                                <td>
                                    ${escapeHtml(
                            String(
                                profile.missing_values ??
                                ""
                            )
                        )}
                                </td>

                                <td>
                                    ${escapeHtml(
                            String(
                                profile.missing_percentage ??
                                ""
                            )
                        )}%
                                </td>

                                <td>
                                    ${escapeHtml(
                            String(
                                profile.mean ??
                                ""
                            )
                        )}
                                </td>

                                <td>
                                    ${escapeHtml(
                            String(
                                profile.median ??
                                ""
                            )
                        )}
                                </td>

                                <td>
                                    ${escapeHtml(
                            String(
                                profile.min ??
                                ""
                            )
                        )}
                                </td>

                                <td>
                                    ${escapeHtml(
                            String(
                                profile.max ??
                                ""
                            )
                        )}
                                </td>

                            </tr>

                        `
                    )
                    .join("");


        } catch (error) {

            console.warn(
                "Profile error:",
                error
            );

        }

    }


    /* ============================================================
       17. QUALITY
       ============================================================ */

    async function fetchQualityReport() {

        try {

            const response =
                await fetch(
                    "/quality"
                );


            if (!response.ok) {
                return;
            }


            const data =
                await response.json();


            const q =
                data.quality;


            const container =
                $("quality-dashboard-content");


            if (
                !container ||
                !q
            ) {
                return;
            }


            const score =
                Number(
                    q.score || 0
                );


            const scoreColor =
                score > 80
                    ? "#22C55E"
                    : "#F59E0B";


            container.innerHTML = `

                <div
                    style="
                        display:flex;
                        gap:2rem;
                        align-items:center;
                        margin-bottom:1.5rem;
                    "
                >

                    <div
                        style="
                            font-size:2.5rem;
                            font-weight:700;
                            color:${scoreColor};
                        "
                    >
                        ${score}/100
                    </div>


                    <div>

                        <h4>
                            Overall Health Status:
                            ${escapeHtml(
                q.status || ""
            )}
                        </h4>


                        <p
                            style="
                                color:#64748B;
                            "
                        >
                            ${escapeHtml(
                q.summary_message ||
                ""
            )}
                        </p>

                    </div>

                </div>


                <h4>
                    Actionable Quality Recommendations
                </h4>


                <ul
                    style="
                        margin-top:.5rem;
                        padding-left:1.25rem;
                        color:#334155;
                    "
                >

                    ${(q.recommendations || [])
                    .map(
                        recommendation =>
                            `
                                    <li
                                        style="
                                            margin-bottom:.35rem;
                                        "
                                    >
                                        ${escapeHtml(
                                recommendation
                            )}
                                    </li>
                                    `
                    )
                    .join("")
                }

                </ul>

            `;


        } catch (error) {

            console.warn(
                "Quality error:",
                error
            );

        }

    }


    /* ============================================================
       18. LOADER
       ============================================================ */

    function showLoader(
        text
    ) {

        if (loaderText) {

            loaderText.textContent =
                text ||
                "Processing...";

        }


        if (loader) {

            loader.classList.remove(
                "hidden"
            );

        }

    }


    function hideLoader() {

        if (loader) {

            loader.classList.add(
                "hidden"
            );

        }

    }


    /* ============================================================
       19. VOICE INPUT
       ============================================================ */

    if (micBtn) {

        micBtn.addEventListener(
            "click",
            async () => {

                /*
                 * Clicking mic while recording = stop.
                 */
                if (
                    mediaRecorder &&
                    mediaRecorder.state ===
                    "recording"
                ) {

                    stopAndTranscribe();

                    return;
                }


                if (
                    !navigator.mediaDevices ||
                    !navigator.mediaDevices.getUserMedia
                ) {

                    showVoiceToast(
                        "🎤 Microphone not supported in this browser."
                    );

                    return;
                }


                try {

                    const stream =
                        await navigator.mediaDevices.getUserMedia(
                            {
                                audio: true
                            }
                        );


                    startRecording(
                        stream
                    );


                } catch (error) {

                    console.error(
                        "Microphone error:",
                        error
                    );


                    if (
                        error.name ===
                        "NotAllowedError" ||
                        error.name ===
                        "PermissionDeniedError"
                    ) {

                        showVoiceToast(
                            "🔒 Microphone permission denied."
                        );

                    } else {

                        showVoiceToast(
                            "⚠️ Could not access microphone."
                        );

                    }

                }

            }
        );

    }


    if (stopRecordingBtn) {

        stopRecordingBtn.addEventListener(
            "click",
            stopAndTranscribe
        );

    }


    function startRecording(
        stream
    ) {

        audioChunks = [];
        recordingSeconds = 0;


        const preferredTypes = [

            "audio/webm;codecs=opus",
            "audio/webm",
            "audio/ogg;codecs=opus",
            "audio/mp4"

        ];


        const mimeType =
            typeof MediaRecorder !==
                "undefined"
                ? (
                    preferredTypes.find(
                        type =>
                            MediaRecorder.isTypeSupported(
                                type
                            )
                    ) || ""
                )
                : "";


        try {

            mediaRecorder =
                mimeType
                    ? new MediaRecorder(
                        stream,
                        {
                            mimeType
                        }
                    )
                    : new MediaRecorder(
                        stream
                    );


        } catch (error) {

            stream
                .getTracks()
                .forEach(
                    track =>
                        track.stop()
                );


            showVoiceToast(
                "⚠️ Voice recording is not supported."
            );


            return;
        }


        mediaRecorder.addEventListener(
            "dataavailable",
            event => {

                if (
                    event.data &&
                    event.data.size > 0
                ) {

                    audioChunks.push(
                        event.data
                    );

                }

            }
        );


        mediaRecorder.addEventListener(
            "stop",
            () => {

                stream
                    .getTracks()
                    .forEach(
                        track =>
                            track.stop()
                    );


                sendAudioToGemini(
                    mimeType ||
                    "audio/webm"
                );

            }
        );


        mediaRecorder.start(
            250
        );


        if (micBtn) {

            micBtn.classList.add(
                "recording"
            );

        }


        if (voiceRecordingBar) {

            voiceRecordingBar.classList.remove(
                "hidden"
            );

        }


        if (voiceTimerEl) {

            voiceTimerEl.textContent =
                "0s";

        }


        showVoiceToast(
            "🎙️ Recording… speak your question"
        );


        recordingTimer =
            setInterval(
                () => {

                    recordingSeconds++;


                    if (voiceTimerEl) {

                        voiceTimerEl.textContent =
                            recordingSeconds +
                            "s";

                    }


                    if (
                        recordingSeconds >=
                        60
                    ) {

                        showVoiceToast(
                            "⏱️ Maximum recording time reached."
                        );


                        stopAndTranscribe();

                    }

                },
                1000
            );

    }


    function stopAndTranscribe() {

        if (
            !mediaRecorder ||
            mediaRecorder.state !==
            "recording"
        ) {
            return;
        }


        clearInterval(
            recordingTimer
        );


        mediaRecorder.stop();


        if (micBtn) {

            micBtn.classList.remove(
                "recording"
            );

            micBtn.classList.add(
                "transcribing"
            );

        }


        if (voiceRecordingBar) {

            voiceRecordingBar.classList.add(
                "hidden"
            );

        }


        showVoiceToast(
            "⏳ Transcribing with Gemini…"
        );

    }


    async function sendAudioToGemini(
        mimeType
    ) {

        if (
            !audioChunks.length
        ) {

            resetMicState();


            showVoiceToast(
                "❌ No audio captured. Please try again."
            );


            return;
        }


        const blob =
            new Blob(
                audioChunks,
                {
                    type: mimeType
                }
            );


        const formData =
            new FormData();


        formData.append(
            "audio",
            blob,
            "recording.webm"
        );


        try {

            const response =
                await fetch(
                    "/transcribe",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (
                response.ok &&
                data.text
            ) {

                if (chatInput) {

                    chatInput.value =
                        data.text;

                    chatInput.focus();


                    chatInput.style.borderColor =
                        "#22C55E";


                    chatInput.style.boxShadow =
                        "0 0 0 3px rgba(34,197,94,.2)";


                    setTimeout(
                        () => {

                            chatInput.style.borderColor =
                                "";

                            chatInput.style.boxShadow =
                                "";

                        },
                        2000
                    );

                }


                showVoiceToast(
                    "✅ Transcribed! Press Send or edit your question."
                );


            } else {

                showVoiceToast(
                    "❌ " +
                    (
                        data.error ||
                        "Transcription failed."
                    )
                );

            }


        } catch (error) {

            console.error(
                "Transcription error:",
                error
            );


            showVoiceToast(
                "❌ Could not reach the transcription server."
            );


        } finally {

            resetMicState();

        }

    }


    function resetMicState() {

        if (micBtn) {

            micBtn.classList.remove(
                "recording",
                "transcribing"
            );

        }


        if (voiceRecordingBar) {

            voiceRecordingBar.classList.add(
                "hidden"
            );

        }


        clearInterval(
            recordingTimer
        );


        mediaRecorder =
            null;

        audioChunks =
            [];

        recordingSeconds =
            0;


        if (voiceTimerEl) {

            voiceTimerEl.textContent =
                "0s";

        }

    }


    function showVoiceToast(
        message,
        durationMs = 2800
    ) {

        let toast =
            $("voice-toast-el");


        if (!toast) {

            toast =
                document.createElement(
                    "div"
                );


            toast.id =
                "voice-toast-el";


            toast.className =
                "voice-toast";


            document.body.appendChild(
                toast
            );

        }


        toast.textContent =
            message;


        toast.classList.add(
            "show"
        );


        clearTimeout(
            toast._hideTimer
        );


        toast._hideTimer =
            setTimeout(
                () => {

                    toast.classList.remove(
                        "show"
                    );

                },
                durationMs
            );

    }


    /* ============================================================
       20. SECURITY HELPER
       ============================================================ */

    function escapeHtml(
        value
    ) {

        if (
            value === null ||
            value === undefined
        ) {
            return "";
        }


        return String(value)

            .replace(
                /&/g,
                "&amp;"
            )

            .replace(
                /</g,
                "&lt;"
            )

            .replace(
                />/g,
                "&gt;"
            )

            .replace(
                /"/g,
                "&quot;"
            )

            .replace(
                /'/g,
                "&#039;"
            );

    }


    /* ============================================================
       21. GLOBAL EXPORTS
       ============================================================ */

    window.fetchDatasetInfo =
        fetchDatasetInfo;

    window.fetchSuggestions =
        fetchSuggestions;

    window.fetchDatasetProfile =
        fetchDatasetProfile;

    window.fetchQualityReport =
        fetchQualityReport;

    window.fetchInsights =
        fetchInsights;

    window.setVisualizationSelection =
        setVisualizationSelection;

    window.renderVisualizationChart =
        renderVisualizationChart;

    window.syncAgentResultToVisualization =
        syncAgentResultToVisualization;

});