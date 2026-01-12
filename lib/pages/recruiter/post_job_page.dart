import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/job_provider.dart';
import '../../widgets/primary_button.dart';

class PostJobPage extends StatefulWidget {
  const PostJobPage({super.key});

  @override
  State<PostJobPage> createState() => _PostJobPageState();
}

class _PostJobPageState extends State<PostJobPage> {
  final titleCtrl = TextEditingController();
  final companyCtrl = TextEditingController();
  final locationCtrl = TextEditingController();
  final categoryCtrl = TextEditingController();
  final salaryCtrl = TextEditingController();
  final descCtrl = TextEditingController();

  bool loading = false;

  @override
  void dispose() {
    titleCtrl.dispose();
    companyCtrl.dispose();
    locationCtrl.dispose();
    categoryCtrl.dispose();
    salaryCtrl.dispose();
    descCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final jobProvider = context.read<JobProvider>();
    return Scaffold(
      resizeToAvoidBottomInset: true,
      appBar: AppBar(title: const Text('Post a job')),
      body: SafeArea(
        top: false,
        child: ListView(
          padding: EdgeInsets.fromLTRB(16, 16, 16, 24 + MediaQuery.of(context).viewInsets.bottom),
          keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
          children: [
            TextField(
              controller: titleCtrl,
              textInputAction: TextInputAction.next,
              decoration: const InputDecoration(labelText: 'Job title'),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: companyCtrl,
              textInputAction: TextInputAction.next,
              decoration: const InputDecoration(labelText: 'Company'),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: locationCtrl,
              textInputAction: TextInputAction.next,
              decoration: const InputDecoration(labelText: 'Location'),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: categoryCtrl,
              textInputAction: TextInputAction.next,
              decoration: const InputDecoration(labelText: 'Category / role type'),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: salaryCtrl,
              textInputAction: TextInputAction.next,
              decoration: const InputDecoration(labelText: 'Salary'),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: descCtrl,
              textInputAction: TextInputAction.newline,
              decoration: const InputDecoration(labelText: 'Description'),
              minLines: 4,
              maxLines: 7,
            ),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: PrimaryButton(
                label: 'Publish role',
                loading: loading,
                onPressed: () async {
                  FocusScope.of(context).unfocus();
                  setState(() => loading = true);
                  try {
                    await jobProvider.postJob(
                      title: titleCtrl.text,
                      company: companyCtrl.text,
                      location: locationCtrl.text,
                      category: categoryCtrl.text,
                      salary: salaryCtrl.text,
                      description: descCtrl.text,
                    );
                    if (mounted) Navigator.pop(context);
                  } catch (error) {
                    if (mounted) _showError(context, error.toString());
                  } finally {
                    if (mounted) setState(() => loading = false);
                  }
                },
              ),
            ),
            // Posted jobs are saved via /jobs POST endpoint. TODO: re-train the recommender when new roles are published.
          ],
        ),
      ),
    );
  }

  void _showError(BuildContext context, String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }
}
