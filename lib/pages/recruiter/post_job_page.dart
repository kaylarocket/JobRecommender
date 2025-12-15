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
      appBar: AppBar(title: const Text('Post a job')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            TextField(controller: titleCtrl, decoration: const InputDecoration(labelText: 'Job title')), const SizedBox(height: 12),
            TextField(controller: companyCtrl, decoration: const InputDecoration(labelText: 'Company')), const SizedBox(height: 12),
            TextField(controller: locationCtrl, decoration: const InputDecoration(labelText: 'Location')), const SizedBox(height: 12),
            TextField(controller: categoryCtrl, decoration: const InputDecoration(labelText: 'Category / role type')), const SizedBox(height: 12),
            TextField(controller: salaryCtrl, decoration: const InputDecoration(labelText: 'Salary')), const SizedBox(height: 12),
            TextField(controller: descCtrl, decoration: const InputDecoration(labelText: 'Description'), maxLines: 6),
            const SizedBox(height: 16),
            PrimaryButton(
              label: 'Publish role',
              loading: loading,
              onPressed: () async {
                setState(() => loading = true);
                await jobProvider.postJob(
                  title: titleCtrl.text,
                  company: companyCtrl.text,
                  location: locationCtrl.text,
                  category: categoryCtrl.text,
                  salary: salaryCtrl.text,
                  description: descCtrl.text,
                );
                setState(() => loading = false);
                if (mounted) Navigator.pop(context);
              },
            ),
            const SizedBox(height: 12),
            const Text('Posted jobs are saved via the /jobs POST endpoint. TODO: re-train the recommender when new roles are published.'),
          ],
        ),
      ),
    );
  }
}
